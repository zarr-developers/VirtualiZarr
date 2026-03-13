from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from zarr.api.asynchronous import open_group as open_group_async
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.storage import ObjectStore

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import (
    validate_and_normalize_path_to_uri,
)

if TYPE_CHECKING:
    import zarr

T = TypeVar("T")
ZarrArrayType = zarr.AsyncArray | zarr.Array


# TODO put these in the enum
ZARR_V2_METADATA_KEYS = (".zarray", ".zattrs", ".zgroup", ".zmetadata")
ZARR_V3_METADATA_KEYS = ("zarr.json",)


class ZarrFormat(Enum):
    """
    Encode all differences between on-disk Zarr formats here.
    
    Note that we still only need to support the zarr-python v3 API, 
    so this enum is only concerned with differences in the native format spec between versions.
    """
    
    V2 = 2
    V3 = 3

    @property
    def scalar_chunk_key_name(self) -> str:
        match self:
            case ZarrFormat.V2:
                return "0"
            case ZarrFormat.V3:
                return "c"
    
    @property
    def possible_metadata_key_names(self) -> tuple[str]:
        match self:
            case ZarrFormat.V2:
                return (".zarray", ".zattrs", ".zgroup", ".zmetadata")
            case ZarrFormat.V3:
                return ("zarr.json",)
            
    # TODO move everything else that depends on format here, instead of the strategy thing


def join_url(base: str, key: str) -> str:
    """Join a base URL (like s3://bucket/store.zarr) with an object key.

    Ensures we don't accidentally produce double slashes (after the scheme)
    and that the returned string is scheme-friendly.
    """
    if not base:
        return key
    # strip trailing slash from base and leading slash from key to avoid '//' in middle
    return base.rstrip("/") + "/" + key.lstrip("/")


def _get_array_name(zarr_array: ZarrArrayType) -> str:
    """Extract and normalize the array name."""
    name = getattr(zarr_array, "name", "") or ""
    return name.lstrip("/")


async def _handle_scalar_array(
    zarr_array: ZarrArrayType, path: str, scalar_key: str
) -> dict[str, dict[str, Any]]:
    """
    Handle scalar arrays (shape == ()).

    Parameters
    ----------
    zarr_array
        The scalar Zarr array.
    path
        Base path for constructing chunk paths.
    scalar_key
        The storage key for the scalar value (e.g., "0" for V2, "c" for V3).

    Returns
    -------
    dict
        Mapping with a single entry for the scalar chunk.
    """
    size = await zarr_array.store.getsize(scalar_key)
    actual_path = join_url(path, scalar_key)
    return {
        "0" if scalar_key == "0" else "c": {
            "path": actual_path,
            "offset": 0,
            "length": size,
        }
    }


async def _build_1d_chunk_mapping(
    zarr_array: ZarrArrayType, path: str, prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build chunk mapping by listing the object store with obstore.

    Uses obstore's list_async with Arrow output to get chunk paths and sizes
    in a single Rust-level call, avoiding per-chunk getsize calls.

    Parameters
    ----------
    zarr_array
        The Zarr array.
    path
        Base path for constructing chunk paths.
    prefix
        Prefix to list and strip from chunk keys.

    Returns
    -------
    Tuple of (stripped_keys, full_paths, sizes) as numpy arrays.
    """
    zarr_format = ZarrFormat(zarr_array.metadata.zarr_format)

    path_batches: list[np.ndarray] = []
    size_batches: list[np.ndarray] = []
    stream = cast(ObjectStore, zarr_array.store).store.list_async(
        prefix=prefix, return_arrow=True
    )
    # TODO is there any purpose to this async for loop? Given that we just join all the results together anyway...
    async for batch in stream:

        # immediately convert to numpy arrays - we can still do efficient operations, and don't need any extra arrow dependencies.
        paths_np = batch.column("path").to_numpy().astype(np.dtypes.StringDType())
        sizes_np = batch.column("size").to_numpy()

        # filter out metadata and directory keys, leaving only valid chunk keys
        # (assumes that there are no other objects inside this directory)
        is_metadata = np.zeros(len(paths_np), dtype=bool)
        for suffix in zarr_format.possible_metadata_key_names:
            is_metadata |= np.strings.endswith(paths_np, suffix)
        is_directory = np.strings.endswith(paths_np, "/")
        chunk_keys_mask = ~(is_metadata | is_directory)
        
        path_batches.append(paths_np[chunk_keys_mask])
        size_batches.append(sizes_np[chunk_keys_mask])

    if not path_batches:
        # no initialized chunks found
        return np.full(0, "", dtype=np.dtypes.StringDType()), np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint64)

    # join batches into one 1D array for all initialized chunks
    all_paths = np.concatenate(path_batches)
    all_sizes = np.concatenate(size_batches)

    # strip the prefix to get chunk keys like "0.0.0"
    stripped_keys = np.strings.slice(all_paths, len(prefix), None)

    # construct full paths
    base = path.rstrip("/") + "/"
    full_paths = np.strings.add(base, all_paths)

    return stripped_keys, full_paths, all_sizes


# TODO get rid of this whole strategy thing - the parser can just be a sequence of functions whose internals dispatch on the version format
class ZarrVersionStrategy(ABC):
    """Abstract base class for handling version-specific Zarr operations."""

    @abstractmethod
    async def get_chunk_mapping(
        self, zarr_array: ZarrArrayType, path: str
    ) -> dict[str, dict[str, Any]]:
        """Get mapping of chunk coordinates to storage locations."""
        ...

    @abstractmethod
    def get_metadata(self, zarr_array: ZarrArrayType) -> ArrayV3Metadata:
        """Get V3 metadata for the array (converting if necessary)."""
        ...

    @abstractmethod
    def get_prefix(self, zarr_array: ZarrArrayType) -> str:
        """Get the storage prefix for chunk listing."""
        ...

    @abstractmethod
    def _get_separator(self, zarr_array: ZarrArrayType) -> str: ...

    @abstractmethod
    def validate(self, zarr_array: ZarrArrayType) -> None:
        """Validate that the array can be virtualized."""


class ZarrV2Strategy(ZarrVersionStrategy):
    """Strategy for handling Zarr V2 arrays."""

    async def get_chunk_mapping(
        self, zarr_array: ZarrArrayType, path: str
    ) -> dict[str, dict[str, Any]]:
        """Create a mapping of chunk coordinates to their storage locations for V2 arrays."""
        name = _get_array_name(zarr_array)
        prefix = f"{name}/" if name else ""

        # Handle scalar arrays
        if zarr_array.shape == ():
            scalar_key = f"{prefix}0"
            return await _handle_scalar_array(zarr_array, path, scalar_key)

        return await _build_1d_chunk_mapping(zarr_array, path, prefix)  # type: ignore[return-value]

    def get_metadata(self, zarr_array: ZarrArrayType) -> ArrayV3Metadata:
        """Convert V2 metadata to V3 format."""

        try:
            from zarr.core.dtype import parse_dtype
            from zarr.metadata.migrate_v3 import _convert_array_metadata
        except (ImportError, AttributeError):
            raise ImportError(
                f"Zarr-Python>=3.1.3 is required for parsing Zarr V2 into Zarr V3. "
                f"Found Zarr version '{zarr.__version__}'"
            )

        v2_metadata = zarr_array.metadata
        assert isinstance(v2_metadata, ArrayV2Metadata)

        if v2_metadata.fill_value is None:
            v2_dict = v2_metadata.to_dict()
            v2_dtype = parse_dtype(cast(Any, v2_dict["dtype"]), zarr_format=2)
            fill_value = v2_dtype.default_scalar()
            v2_dict["fill_value"] = v2_dtype.to_json_scalar(fill_value, zarr_format=2)
            temp_v2 = ArrayV2Metadata.from_dict(v2_dict)
            v3_metadata = _convert_array_metadata(temp_v2)
        else:
            # Normal conversion; allow other errors to propagate.
            v3_metadata = _convert_array_metadata(v2_metadata)

        # Set dimension names from attributes or generate defaults
        if v3_metadata.dimension_names is None:
            v3_dict = v3_metadata.to_dict()
            dim_names = None
            if hasattr(v2_metadata, "attributes") and v2_metadata.attributes:
                dim_names = v2_metadata.attributes.get("_ARRAY_DIMENSIONS")

            if dim_names:
                v3_dict["dimension_names"] = dim_names
            else:
                array_name = zarr_array.name.lstrip("/") if zarr_array.name else "array"
                v3_dict["dimension_names"] = [
                    f"{array_name}_dim_{i}" for i in range(len(zarr_array.shape))
                ]
            v3_metadata = ArrayV3Metadata.from_dict(v3_dict)

        v3_dict = v3_metadata.to_dict()

        # Replace V2ChunkKeyEncoding with V3 DefaultChunkKeyEncoding
        # The automatic conversion preserves V2's encoding, causing zarr to use V2-style
        # paths (array/0) instead of V3-style (array/c/0). This ensures V3 semantics.
        if (
            "attributes" in v3_dict
            and isinstance(v3_dict["attributes"], dict)
            and "_ARRAY_DIMENSIONS" in v3_dict["attributes"]
        ):
            del v3_dict["attributes"]["_ARRAY_DIMENSIONS"]
            v3_metadata = ArrayV3Metadata.from_dict(v3_dict)
        v3_dict["chunk_key_encoding"] = {"name": "default", "separator": "."}
        v3_metadata = ArrayV3Metadata.from_dict(v3_dict)

        return v3_metadata

    def get_prefix(self, zarr_array: ZarrArrayType) -> str:
        name = _get_array_name(zarr_array)
        return f"{name}/" if name else ""

    def _get_separator(self, zarr_array: ZarrArrayType) -> str:
        from typing import cast

        return cast(ArrayV2Metadata, zarr_array.metadata).dimension_separator

    def validate(self, zarr_array: ZarrArrayType) -> None:
        pass  # no restrictions for V2


# TODO rename this strategy thing
class ZarrV3Strategy(ZarrVersionStrategy):
    """Strategy for handling Zarr V3 arrays."""

    # TODO: There's nothing really format-specific in here
    # In fact this is not even used
    async def get_chunk_mapping(
        self, zarr_array: ZarrArrayType, path: str
    ) -> dict[str, dict[str, Any]]:
        """Create a mapping of chunk coordinates to their storage locations for V3 arrays."""
        # Check for sharding - not yet supported
        from zarr.codecs import ShardingCodec

        # Type narrowing: V3 strategy only handles V3 arrays with V3 metadata
        metadata = zarr_array.metadata
        if not isinstance(metadata, ArrayV3Metadata):
            raise TypeError(
                f"Expected ArrayV3Metadata in V3 strategy, got {type(metadata)}"
            )

        if any(isinstance(codec, ShardingCodec) for codec in metadata.codecs):
            raise NotImplementedError(
                "Zarr V3 arrays with sharding are not yet supported. "
                "Sharding stores multiple chunks in a single storage object with non-zero offsets, "
                "which VirtualiZarr does not currently handle. "
                "Reading sharded arrays without proper offset handling would result in corrupted data."
            )

        name = _get_array_name(zarr_array)

        # TODO this logic is basically common to both formats
        # Handle scalar arrays
        if zarr_array.shape == ():
            scalar_key = f"{name}/c" if name else "c"
            return await _handle_scalar_array(zarr_array, path, scalar_key)

        # List chunk keys under the c/ subdirectory
        prefix = f"{name}/c/" if name else "c/"
        return await _build_1d_chunk_mapping(zarr_array, path, prefix)  # type: ignore[return-value]

    def get_metadata(self, zarr_array: ZarrArrayType) -> ArrayV3Metadata:
        """Return V3 metadata as-is (no conversion needed)."""
        return zarr_array.metadata  # type: ignore[return-value]

    def get_prefix(self, zarr_array: ZarrArrayType) -> str:
        name = _get_array_name(zarr_array)
        return f"{name}/c/" if name else "c/"

    def _get_separator(self, zarr_array: ZarrArrayType) -> str:
        from typing import cast

        metadata = cast(ArrayV3Metadata, zarr_array.metadata)
        return cast(DefaultChunkKeyEncoding, metadata.chunk_key_encoding).separator

    def validate(self, zarr_array: ZarrArrayType) -> None:
        from zarr.codecs import ShardingCodec

        if not isinstance(zarr_array.metadata, ArrayV3Metadata):
            # TODO: why does this return instead of raising?
            return
        
        # TODO: There don't need to be any other checks for sharding aside from this
        if any(
            isinstance(codec, ShardingCodec) for codec in zarr_array.metadata.codecs
        ):
            raise NotImplementedError(
                "Zarr V3 arrays with sharding are not yet supported. "
                "Sharding stores multiple chunks in a single storage object with non-zero offsets, "
                "which VirtualiZarr does not currently handle. "
                "Reading sharded arrays without proper offset handling would result in corrupted data."
            )


def get_strategy(zarr_array: ZarrArrayType) -> ZarrVersionStrategy:
    """
    Factory function to get the appropriate strategy for a Zarr array.

    Parameters
    ----------
    zarr_array
        The Zarr array to get a strategy for.

    Returns
    -------
    ZarrVersionStrategy
        The appropriate strategy instance for the array's Zarr format version.

    Raises
    ------
    NotImplementedError
        If the Zarr format version is not supported.
    """
    zarr_format = zarr_array.metadata.zarr_format
    if zarr_format == 2:
        return ZarrV2Strategy()
    elif zarr_format == 3:
        return ZarrV3Strategy()
    else:
        raise NotImplementedError(f"Zarr format {zarr_format} is not supported")


async def build_chunk_manifest(zarr_array: ZarrArrayType, path: str) -> ChunkManifest:
    """Build a ChunkManifest from chunk coordinate mappings.

    Note: Chunk keys are discovered by listing what's actually in storage rather than
    generating all possible keys from the chunk grid. Zarr allows chunks to be missing
    (sparse arrays), and VirtualiZarr manifests preserve this sparsity. When chunks are
    missing, Zarr will return the fill_value for those regions when the array is read.
    """
    strategy = get_strategy(zarr_array)
    strategy.validate(zarr_array)
    chunk_grid_shape = zarr_array.cdata_shape

    prefix = strategy.get_prefix(zarr_array)

    # Handle scalar arrays
    if zarr_array.shape == ():

        # TODO simplify the construction of the enum here?
        scalar_key = ZarrFormat(zarr_array.metadata.zarr_format).scalar_chunk_key_name

        # Can only contain a single chunk, so just GET that instead of LISTing a whole directory unnecessarily
        # TODO what if the single chunk is uninitialized? Zarr technically allows that possibility doesn't it?
        size = await zarr_array.store.getsize(prefix + scalar_key)
        actual_path = join_url(path, scalar_key)

        return ChunkManifest(
            {
                # TODO: is this correct? Shouldn't all ChunkManifests not depend on the format they were created from?
                "0" if scalar_key == "0" else "c": {
                    "path": actual_path,
                    "offset": 0,
                    "length": size,
                }
            }
        )

    stripped_keys, full_paths, all_lengths = await _build_1d_chunk_mapping(zarr_array, path, prefix)

    # TODO exit early in empty array case?
    if len(stripped_keys) == 0:
        raise NotImplementedError

    total_size = zarr_array.nchunks
    separator = strategy._get_separator(zarr_array)

    # split "0.0.0" style keys into per-dimension integer coords
    # TODO replace np.char.split with np.strings.split once it exists
    split_keys = np.char.split(stripped_keys, sep=separator)
    coords = np.array(
        [[int(c) for c in key] for key in split_keys], dtype=np.int64
    ).T  # shape: (ndim, nchunks)
    flat_positions = np.ravel_multi_index(coords, chunk_grid_shape)

    # scatter listed chunks into dense flat arrays (empty string / 0 = missing)
    dense_paths = np.full(total_size, "", dtype=np.dtypes.StringDType())
    dense_lengths = np.zeros(total_size, dtype=np.uint64)
    dense_offsets = np.zeros(total_size, dtype=np.uint64)

    dense_paths[flat_positions] = full_paths
    dense_lengths[flat_positions] = all_lengths

    return ChunkManifest.from_arrays(
        paths=dense_paths.reshape(chunk_grid_shape),
        offsets=dense_offsets.reshape(chunk_grid_shape),
        lengths=dense_lengths.reshape(chunk_grid_shape),
    )


def get_metadata(zarr_array: ZarrArrayType) -> ArrayV3Metadata:
    """
    Get V3 metadata for an array, converting from V2 if necessary.

    Parameters
    ----------
    zarr_array
        The Zarr array to get metadata for.

    Returns
    -------
    ArrayV3Metadata
        V3 metadata for the array.
    """
    strategy = get_strategy(zarr_array)
    return strategy.get_metadata(zarr_array)


async def _construct_manifest_array(
    zarr_array: zarr.AsyncArray[Any], path: str
) -> ManifestArray:
    """Construct a ManifestArray from a zarr array."""
    array_metadata = get_metadata(zarr_array)
    chunk_manifest = await build_chunk_manifest(zarr_array, path)
    return ManifestArray(metadata=array_metadata, chunkmanifest=chunk_manifest)


async def _construct_manifest_group(
    path: str,
    store: zarr.storage.ObjectStore,
    *,
    skip_variables: str | Iterable[str] | None = None,
    group: str | None = None,
) -> ManifestGroup:
    """Construct a ManifestGroup from a zarr group."""
    zarr_group = await open_group_async(store=store, path=group, mode="r")

    zarr_array_keys = [key async for key in zarr_group.array_keys()]
    _skip_variables = [] if skip_variables is None else list(skip_variables)

    zarr_arrays = await asyncio.gather(
        *[
            zarr_group.getitem(var)
            for var in zarr_array_keys
            if var not in _skip_variables
        ]
    )

    manifest_arrays = await asyncio.gather(
        *[_construct_manifest_array(array, path) for array in zarr_arrays]  # type: ignore[arg-type]
    )

    manifest_dict = {
        array.basename: result for array, result in zip(zarr_arrays, manifest_arrays)
    }

    manifest_group = ManifestGroup(manifest_dict, attributes=zarr_group.attrs)
    # TODO: This is weird - why isn't this just set in the ManifestGroup constructor?
    manifest_group._metadata = GroupMetadata(
        attributes=dict(zarr_group.attrs) if zarr_group.attrs is not None else {},
        zarr_format=3,
        consolidated_metadata=None,
    )

    return manifest_group


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine, handling the case where an event loop is already running.

    In environments like Jupyter notebooks, an event loop is already running,
    so ``asyncio.run()`` raises ``RuntimeError``. In that case we run the
    coroutine in a separate thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – the simple path.
        return asyncio.run(coro)

    # A loop is already running (e.g. Jupyter).  Execute in a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class ZarrParser:
    """
    Parser for creating virtual references to existing Zarr stores.

    The ZarrParser creates lightweight virtual references to chunks in existing
    Zarr stores without copying data. It supports both Zarr V2 and V3 formats,
    automatically converting V2 metadata to V3 format.

    Parameters
    ----------
    group : str, optional
        Path to a specific group within the Zarr store to use as the root.
        Uses forward slashes for nested groups (e.g., "model/output").
        Default is None, which uses the store's root group.
    skip_variables : iterable of str, optional
        Names of variables (arrays) to exclude when creating the virtual store.
        Useful for filtering out auxiliary data or large variables that aren't
        needed. Default is None, which includes all variables.

    Attributes
    ----------
    group : str or None
        The group path to use as root.
    skip_variables : iterable of str or None
        Variables to exclude from virtualization.

    Methods
    -------
    __call__(url, registry)
        Create a virtual representation of a Zarr store.

    See Also
    --------
    virtualizarr.open_virtual_dataset : High-level function for opening virtual datasets.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the `__call__` method.

        Parameters
        ----------
        group : str | None, optional (default: None)
            The group within the original Zarr store to be used as the root group for
            the ManifestStore (default: the Zarr store's root group).
        skip_variables : Iterable[str] | None, optional (default: None)
            Variables in the Zarr store that will be ignored when creating the
            `ManifestStore` (default: None, do not ignore any variables).
        """
        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given Zarr store to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        url : str
            URL or path to the Zarr store. Supports various protocols:

            - Local filesystem: "file:///path/to/store.zarr" or "/path/to/store.zarr"
            - S3: "s3://bucket/path/to/store.zarr"
            - Google Cloud Storage: "gs://bucket/path/to/store.zarr"
            - Azure Blob Storage: "az://container/path/to/store.zarr"
            - HTTP/HTTPS: "https://example.com/store.zarr"

        registry : ObjectStoreRegistry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for
            resolving urls and reading data.

        Returns
        -------
        [ManifestStore][virtualizarr.manifests.ManifestStore]
            A virtual representation of the Zarr store with references to
            the original chunk locations.

        Raises
        ------
        ValueError
            If the URL cannot be resolved or normalized.
        KeyError
            If the registry doesn't contain an appropriate store for the URL.
        NotImplementedError
            If the Zarr store uses an unsupported format version.

        See Also
        --------
        virtualizarr.open_virtual_dataset : High-level interface for virtual datasets.
        virtualizarr.manifests.ManifestStore : The returned virtual store object.
        """
        uri = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())

        object_store, store_relative_path = registry.resolve(uri)
        zarr_store = ObjectStore(store=object_store)  # type: ignore[type-var]

        # Compute the store root URI by stripping the relative path from the full URI
        rel_path = str(store_relative_path)
        store_root_uri = uri.removesuffix(rel_path).rstrip("/") if rel_path else uri

        # Combine the store-relative path with optional group to get the full
        # path within the object store to the zarr group
        group_path = rel_path
        if self.group:
            group_path = f"{group_path}/{self.group}" if group_path else self.group
        
        # Parse groups recursively from the root, concurrently
        coro = _construct_manifest_group(
            store=zarr_store,
            path=store_root_uri,
            group=group_path or None,
            skip_variables=self.skip_variables,
        )
        manifest_group = _run_async(coro)

        return ManifestStore(registry=registry, group=manifest_group)
