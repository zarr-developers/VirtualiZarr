from __future__ import annotations

import asyncio
import concurrent.futures
import math
from collections.abc import Coroutine, Iterable
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import obstore
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from zarr.api.asynchronous import open_group as open_group_async
from zarr.codecs import ShardingCodec
from zarr.core.chunk_grids import RegularChunkGrid
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
from virtualizarr.manifests.utils import ChunkKeySeparator
from virtualizarr.utils import determine_chunk_grid_shape

# obstore doesn't export a public base type for stores, so we use Any for now.
ObstoreStore = Any

T = TypeVar("T")


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


def join_url(base: str, key: str) -> str:
    """Join a base URL (like s3://bucket/store.zarr) with an object key.

    Ensures we don't accidentally produce double slashes (after the scheme)
    and that the returned string is scheme-friendly.
    """
    if not base:
        return key
    # strip trailing slash from base and leading slash from key to avoid '//' in middle
    return base.rstrip("/") + "/" + key.lstrip("/")


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

    # This is the only restriction on what Zarr Arrays cannot be virtualized
    if any(isinstance(codec, ShardingCodec) for codec in array_v3_metadata.codecs):
        raise NotImplementedError(
            f"Zarr V3 arrays with sharding are not yet supported, but array {path} uses the ShardingCodec."
            "Sharding stores multiple chunks in a single storage object with non-zero offsets, "
            "which VirtualiZarr does not currently handle. "
            "Reading sharded arrays without proper offset handling would result in corrupted data."
        )

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
    chunk_manifest = await build_chunk_manifest(
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
    if v3_dict.get("dimension_names") is None and dim_names:
        v3_dict["dimension_names"] = dim_names

    # _ARRAY_DIMENSIONS is a V2 convention that gets promoted to dimension_names in V3,
    # so remove it from attributes to avoid duplication.
    if "_ARRAY_DIMENSIONS" in attrs:
        del attrs["_ARRAY_DIMENSIONS"]

    return v3_dict


async def build_chunk_manifest(
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

    chunk_grid_shape = determine_chunk_grid_shape(
        metadata.shape, cast(RegularChunkGrid, metadata.chunk_grid).chunk_shape
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
