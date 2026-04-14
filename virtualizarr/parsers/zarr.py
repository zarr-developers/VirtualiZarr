from __future__ import annotations

import asyncio
import concurrent.futures
import math
from abc import ABC, abstractmethod
from collections.abc import Coroutine, Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import obstore
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from zarr.api.asynchronous import open_group as open_group_async
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.core.metadata.v3 import RegularChunkGrid
from zarr.experimental import ChunkGrid
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
from virtualizarr.manifests.utils import ChunkKeySeparator

if TYPE_CHECKING:
    import pyarrow as pa

# obstore doesn't export a public base type for stores, so we use Any for now.
ObstoreStore = Any

T = TypeVar("T")
ZarrArrayType = zarr.AsyncArray | zarr.Array


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


def _normalize_chunk_keys(chunk_keys: list[str], prefix: str) -> list[str]:
    """
    Normalize chunk keys to dot-separated coordinates.

    Strips the prefix from each key and replaces '/' with '.' for coordinate notation.
    """
    chunk_coords = [
        k[len(prefix) :] if prefix and k.startswith(prefix) else k for k in chunk_keys
    ]
    return [coord.replace("/", ".") for coord in chunk_coords]


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


async def _build_chunk_mapping(
    zarr_array: ZarrArrayType, path: str, prefix: str
) -> tuple["pa.Array", "pa.Array", "pa.Array"] | None:
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
    Tuple of (normalized_keys, full_paths, sizes) as PyArrow arrays, or None if no chunks found.
    """
    import pyarrow as pa  # type: ignore[import-untyped,import-not-found]
    import pyarrow.compute as pc  # type: ignore[import-untyped,import-not-found]

    path_batches = []
    size_batches = []
    stream = cast(ObjectStore, zarr_array.store).store.list_async(
        prefix=prefix, return_arrow=True
    )
    async for batch in stream:
        pa_path_col = pa.array(batch.column("path"))
        not_metadata = pc.invert(
            pc.or_(
                pc.match_substring(pa_path_col, pattern="/."),
                pc.starts_with(pa_path_col, "."),
            )
        )

        filtered_paths = pa_path_col.filter(not_metadata)
        filtered_sizes = pa.array(batch.column("size")).filter(not_metadata)
        path_batches.append(filtered_paths)
        size_batches.append(filtered_sizes)

    if not path_batches:
        return None

    all_paths = pa.concat_arrays(path_batches)
    all_sizes = pa.concat_arrays(size_batches)

    if len(all_paths) == 0:
        return None
    stripped_keys = pc.utf8_replace_slice(
        all_paths, start=0, stop=len(prefix), replacement=""
    )

    # construct full paths
    full_paths = pc.binary_join_element_wise(
        pa.scalar(path.rstrip("/")), all_paths, "/"
    )

    return stripped_keys, full_paths, all_sizes


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

        return await _build_chunk_mapping(zarr_array, path, prefix)  # type: ignore[return-value]

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


class ZarrV3Strategy(ZarrVersionStrategy):
    """Strategy for handling Zarr V3 arrays."""

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

        # Handle scalar arrays
        if zarr_array.shape == ():
            scalar_key = f"{name}/c" if name else "c"
            return await _handle_scalar_array(zarr_array, path, scalar_key)

        # List chunk keys under the c/ subdirectory
        prefix = f"{name}/c/" if name else "c/"
        return await _build_chunk_mapping(zarr_array, path, prefix)  # type: ignore[return-value]

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
            return
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
    import pyarrow as pa  # type: ignore[import-untyped,import-not-found]
    import pyarrow.compute as pc  # type: ignore[import-untyped,import-not-found]

    strategy = get_strategy(zarr_array)
    strategy.validate(zarr_array)
    chunk_grid_shape = ChunkGrid.from_metadata(zarr_array.metadata).grid_shape

    if zarr_array.shape == ():
        chunk_map = await strategy.get_chunk_mapping(zarr_array, path)
        if not chunk_map:
            return ChunkManifest(chunk_map, shape=chunk_grid_shape)
        entry = next(iter(chunk_map.values()))
        return ChunkManifest._from_arrow(
            paths=pa.array([entry["path"]], type=pa.string()),
            offsets=pa.array([entry["offset"]], type=pa.uint64()),
            lengths=pa.array([entry["length"]], type=pa.uint64()),
            shape=chunk_grid_shape,
        )

    prefix = strategy.get_prefix(zarr_array)

    result = await _build_chunk_mapping(zarr_array, path, prefix)

    if result is None:
        return ChunkManifest({}, shape=chunk_grid_shape)

    stripped_keys, full_paths, all_lengths = result

    total_size = zarr_array.nchunks
    separator = strategy._get_separator(zarr_array)
    split_keys = pc.split_pattern(stripped_keys, pattern=separator)
    coords = [
        pc.cast(pc.list_element(split_keys, dim), pa.int64()).to_numpy()
        for dim in range(zarr_array.ndim)
    ]
    flat_positions = pa.array(np.ravel_multi_index(coords, chunk_grid_shape))

    # scatter listed chunks into a dense flat array (nulls = missing chunks)
    updates = pa.table(
        {"idx": flat_positions, "path": full_paths, "length": all_lengths}
    )
    dense = (
        pa.table({"idx": pa.array(np.arange(total_size, dtype=np.int64))})
        .join(updates, "idx", join_type="left outer")
        .sort_by("idx")
    )

    return ChunkManifest._from_arrow(
        paths=dense["path"].combine_chunks(),
        offsets=pa.repeat(pa.scalar(0, type=pa.uint64()), total_size),
        lengths=dense["length"].combine_chunks(),
        shape=chunk_grid_shape,
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


class ZarrFormat(Enum):
    """
    Encode all differences between on-disk Zarr formats here.

    Note that we still only need to support the zarr-python v3 API,
    so this enum is only concerned with differences in the native format spec between versions.
    """

    V2 = 2
    V3 = 3

    @property
    def metadata_key_names(self) -> tuple[str, ...]:
        match self:
            case ZarrFormat.V2:
                return (".zarray", ".zattrs", ".zgroup", ".zmetadata")
            case ZarrFormat.V3:
                return ("zarr.json",)

    @property
    def scalar_chunk_key_name(self) -> str:
        match self:
            case ZarrFormat.V2:
                return "0"
            case ZarrFormat.V3:
                return "c"

    @property
    def chunks_dir_prefix(self) -> str:
        match self:
            case ZarrFormat.V2:
                return ""
            case ZarrFormat.V3:
                return "c/"


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
        coro = construct_manifest_group(
            store=zarr_store,
            path=store_root_uri,
            group=group_path or None,
            skip_variables=self.skip_variables,
        )
        manifest_group = _run_async(coro)

        return ManifestStore(registry=registry, group=manifest_group)


async def construct_manifest_group(
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
        *[construct_manifest_array(array, path) for array in zarr_arrays]  # type: ignore[arg-type]
    )

    manifest_dict = {
        array.basename: result for array, result in zip(zarr_arrays, manifest_arrays)
    }

    return ManifestGroup(manifest_dict, attributes=zarr_group.attrs)


async def construct_manifest_array(
    zarr_array: zarr.AsyncArray[Any], path: str
) -> ManifestArray:
    """Construct a ManifestArray from a zarr array."""
    array_v3_metadata = metadata_as_v3(zarr_array.metadata)

    if not isinstance(array_v3_metadata.chunk_grid, RegularChunkGrid):
        raise NotImplementedError(
            f"Only RegularChunkGrid is supported, but array {zarr_array.path} "
            f"uses {type(array_v3_metadata.chunk_grid).__name__}."
        )

    # The on-disk format determines how chunks are stored (e.g. V2 has no c/ prefix),
    # which differs from the always-V3 metadata we use internally.
    on_disk_zarr_format = ZarrFormat(zarr_array.metadata.zarr_format)
    on_disk_separator: ChunkKeySeparator = (
        zarr_array.metadata.chunk_key_encoding.separator
        if on_disk_zarr_format == ZarrFormat.V3
        else "."
    )

    obs_store = cast(ObjectStore, zarr_array.store).store
    chunk_manifest = await _build_chunk_manifest_from_store(
        obs_store=obs_store,
        array_path=zarr_array.path,
        store_base_uri=path,
        metadata=array_v3_metadata,
        on_disk_zarr_format=on_disk_zarr_format,
        on_disk_separator=on_disk_separator,
    )

    return ManifestArray(metadata=array_v3_metadata, chunkmanifest=chunk_manifest)


def metadata_as_v3(metadata: ArrayV3Metadata | ArrayV2Metadata) -> ArrayV3Metadata:
    """Convert metadata to V3 format with normalized chunk_key_encoding."""

    if isinstance(metadata, ArrayV2Metadata):
        v3_dict = _convert_v2_to_v3_dict(metadata)
    else:
        v3_dict = metadata.to_dict()

    # Normalize chunk_key_encoding to DefaultChunkKeyEncoding with "." separator.
    # The ManifestStore expects dot-separated keys (e.g. "0.0.0"), so we enforce
    # this regardless of what the on-disk store uses.
    v3_dict["chunk_key_encoding"] = {"name": "default", "separator": "."}
    return ArrayV3Metadata.from_dict(v3_dict)


def _convert_v2_to_v3_dict(metadata: ArrayV2Metadata) -> dict:
    """Convert V2 metadata to a V3 dict, handling fill_value, dimensions, and attributes."""

    try:
        from zarr.core.dtype import parse_dtype
        from zarr.metadata.migrate_v3 import _convert_array_metadata
    except (ImportError, AttributeError):
        raise ImportError(
            f"Zarr-Python>=3.1.3 is required for parsing Zarr V2 into Zarr V3. "
            f"Found Zarr version '{zarr.__version__}'"
        )

    # V3 requires a non-None fill_value, but V2 allows it. If missing, set to the
    # dtype's default (e.g. 0 for int) before converting. We roundtrip through a dict
    # because ArrayV2Metadata is immutable.
    if metadata.fill_value is None:
        v2_dict = metadata.to_dict()
        v2_dtype = parse_dtype(cast(Any, v2_dict["dtype"]), zarr_format=2)
        fill_value = v2_dtype.default_scalar()
        v2_dict["fill_value"] = v2_dtype.to_json_scalar(fill_value, zarr_format=2)
        metadata = ArrayV2Metadata.from_dict(v2_dict)

    v3_dict = _convert_array_metadata(metadata).to_dict()

    # _convert_array_metadata doesn't promote V2's _ARRAY_DIMENSIONS attribute
    # to V3's dimension_names, so we do it manually.
    attrs = cast(dict, v3_dict.get("attributes", {}))
    dim_names = attrs.get("_ARRAY_DIMENSIONS")
    if v3_dict.get("dimension_names") is None and dim_names is not None:
        v3_dict["dimension_names"] = dim_names

    # _ARRAY_DIMENSIONS is a V2 convention that gets promoted to dimension_names in V3,
    # so remove it from attributes to avoid duplication.
    if "_ARRAY_DIMENSIONS" in attrs:
        del attrs["_ARRAY_DIMENSIONS"]

    return v3_dict


async def _build_chunk_manifest_from_store(
    obs_store: ObstoreStore,
    array_path: str,
    store_base_uri: str,
    metadata: ArrayV3Metadata,
    on_disk_zarr_format: ZarrFormat,
    on_disk_separator: ChunkKeySeparator,
) -> ChunkManifest:
    """Build a ChunkManifest from chunk coordinate mappings.

    Parameters
    ----------
    obs_store
        The obstore ObjectStore for accessing chunk data.
    array_path
        The array's path within the store (e.g. "air" or "group/air").
    store_base_uri
        The base URI of the store (e.g. "s3://bucket/store.zarr").
    metadata
        V3 metadata for the array.
    on_disk_zarr_format
        The actual on-disk zarr format version (may differ from ``metadata.zarr_format``
        which is always 3 after conversion).
    on_disk_separator
        The chunk key separator used on disk (e.g. ``"."`` or ``"/"``).

    Notes
    -----
    Chunk keys are discovered by listing what's actually in storage rather than
    generating all possible keys from the chunk grid. Zarr allows chunks to be missing
    (sparse arrays), and VirtualiZarr manifests preserve this sparsity. When chunks are
    missing, Zarr will return the fill_value for those regions when the array is read.
    """

    # For sharded arrays, chunk_grid.chunk_shape is the shard shape (not the inner
    # chunk shape, which lives inside the ShardingCodec config). So this grid describes
    # the number of shard files on disk, which is exactly what we want for the manifest.
    chunk_shape = cast(RegularChunkGrid, metadata.chunk_grid).chunk_shape
    chunk_grid_shape = tuple(
        math.ceil(s / c) for s, c in zip(metadata.shape, chunk_shape)
    )
    total_size = math.prod(chunk_grid_shape)

    # Handle scalar arrays
    if metadata.shape == ():
        # Can only contain a single chunk, so just GET that instead of LISTing a whole directory unnecessarily
        scalar_key = on_disk_zarr_format.scalar_chunk_key_name
        store_key = join_url(array_path, scalar_key)

        try:
            head = await obstore.head_async(obs_store, store_key)
        except (FileNotFoundError, obstore.exceptions.NotFoundError):
            # technically I think the zarr spec allows for the possibility that the scalar chunk is uninitialized
            return ChunkManifest({})

        size = head["size"]
        full_path = join_url(store_base_uri, store_key)
        return ChunkManifest(
            {
                "c": {
                    "path": full_path,
                    "offset": 0,
                    "length": size,
                }
            }
        )

    # Build 1d array of all initialized chunk paths and their lengths
    nonscalar_chunks_prefix = join_url(
        array_path, on_disk_zarr_format.chunks_dir_prefix
    )
    stripped_keys, full_paths, all_lengths = await build_1d_chunk_mapping(
        obs_store, store_base_uri, nonscalar_chunks_prefix, on_disk_zarr_format
    )

    if len(stripped_keys) == 0:
        # No initialized chunks found, so manifest is empty, and we can exit early.
        return ChunkManifest({}, shape=chunk_grid_shape)

    # split "0.0.0" style keys into per-dimension integer coords
    # TODO replace np.char.split with np.strings.split once it exists
    split_keys = np.char.split(stripped_keys, sep=on_disk_separator)
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


async def build_1d_chunk_mapping(
    obs_store: ObstoreStore,
    store_base_uri: str,
    array_chunks_prefix: str,
    zarr_format: ZarrFormat,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build chunk mapping by listing the object store with obstore.

    Uses obstore's list_async with Arrow output to get chunk paths and sizes
    in a single Rust-level call, avoiding per-chunk getsize calls.

    Parameters
    ----------
    obs_store
        The obstore ObjectStore to list.
    store_base_uri
        The base URI of the store (e.g. "s3://bucket/store.zarr"), used to construct full chunk paths.
    array_chunks_prefix
        Store-relative prefix to list and strip from chunk keys (e.g. "air/c/").
    zarr_format
        The zarr format version.

    Returns
    -------
    Tuple of (stripped_keys, full_paths, sizes) as numpy arrays.
    """
    path_batches: list[np.ndarray] = []
    size_batches: list[np.ndarray] = []
    stream = obs_store.list_async(prefix=array_chunks_prefix, return_arrow=True)
    async for batch in stream:
        # Immediately convert to numpy arrays - we can still do efficient manipulations, and don't need any extra arrow dependencies.
        # Note: The .astype is only needed because .to_numpy converts to a numpy object array of python `str` objects, which is inefficient.
        # TODO: Change this if arrow -> numpy support for variable length strings ever improves, see https://github.com/zarr-developers/VirtualiZarr/issues/922#issuecomment-4051049630
        paths_np = batch.column("path").to_numpy().astype(np.dtypes.StringDType())
        sizes_np = batch.column("size").to_numpy()

        # filter out metadata and directory keys, leaving only valid chunk keys
        # (assumes that there are no other objects inside this directory)
        is_metadata = np.zeros(len(paths_np), dtype=bool)
        for suffix in zarr_format.metadata_key_names:
            is_metadata |= np.strings.endswith(paths_np, suffix)
        is_directory = np.strings.endswith(paths_np, "/")
        chunk_keys_mask = ~(is_metadata | is_directory)

        path_batches.append(paths_np[chunk_keys_mask])
        size_batches.append(sizes_np[chunk_keys_mask])

    if not path_batches:
        # no initialized chunks found
        return (
            np.full(0, "", dtype=np.dtypes.StringDType()),
            np.zeros(0, dtype=np.uint64),
            np.zeros(0, dtype=np.uint64),
        )

    # join batches into one 1D array for all initialized chunks
    all_paths = np.concatenate(path_batches)
    all_sizes = np.concatenate(size_batches)

    # strip the prefix to get chunk keys like "0.0.0"
    # TODO: replace with np.strings.slice once minimum numpy is 2.3.0
    stripped_keys = np.strings.replace(all_paths, array_chunks_prefix, "", 1)

    # construct full URIs for each chunk
    full_paths = np.strings.add(store_base_uri + "/", all_paths)

    return stripped_keys, full_paths, all_sizes
