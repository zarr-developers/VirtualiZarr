from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import obstore
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from zarr.core.metadata import ArrayV3Metadata
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
from virtualizarr.parsers.utils import construct_manifest_group_tree
from virtualizarr.parsers.zarr.common import (
    ObstoreStore,
    RegularChunkGridMetadata,
    ZarrFormat,
    _run_async,
    chunk_entries_to_manifest,
    join_url,
    parse_array_layout,
)
from virtualizarr.utils import determine_chunk_grid_shape


class ZarrParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an existing Zarr store.

    Creates lightweight virtual references to chunks in an existing Zarr store
    without copying data. Supports both Zarr V2 and V3 formats, automatically
    converting V2 metadata to V3.

    Parameters
    ----------
    group
        Path to a specific group within the Zarr store to be used as the Zarr
        root group for the ManifestStore. Uses forward slashes for nested
        groups (e.g., "model/output"). Default is None, which uses the store's
        root group.
    skip_variables
        Variables in the Zarr store that will be ignored when creating the
        ManifestStore. Default is None, which includes all variables.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
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
    """Construct a ManifestGroup from a zarr group, recursing into subgroups."""
    return await construct_manifest_group_tree(
        store,
        build_manifest_array=lambda array: construct_manifest_array(array, path),
        group=group,
        skip_variables=skip_variables,
    )


async def construct_manifest_array(
    zarr_array: zarr.AsyncArray[Any], path: str
) -> ManifestArray:
    """Construct a ManifestArray from a zarr array."""
    array_v3_metadata, on_disk_zarr_format, on_disk_separator, _ = parse_array_layout(
        zarr_array
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

    # For sharded arrays, chunk_grid.chunk_shape is the shard shape (not the inner
    # chunk shape, which lives inside the ShardingCodec config). So this grid describes
    # the number of shard files on disk, which is exactly what we want for the manifest.
    chunk_grid_shape = determine_chunk_grid_shape(
        metadata.shape, cast(RegularChunkGridMetadata, metadata.chunk_grid).chunk_shape
    )

    # Handle scalar arrays
    if metadata.shape == ():
        # Can only contain a single chunk, so just GET that instead of LISTing a whole directory unnecessarily
        scalar_key = on_disk_zarr_format.scalar_chunk_key_name
        store_key = join_url(array_path, scalar_key)

        try:
            head = await obstore.head_async(obs_store, store_key)
        except (FileNotFoundError, obstore.exceptions.NotFoundError):
            # The zarr spec allows the scalar chunk to be uninitialized (e.g. CF
            # grid-mapping / CRS variables carry only attributes). An empty
            # manifest still needs its (empty) chunk grid shape.
            return ChunkManifest({}, shape=chunk_grid_shape)

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

    # Build 1d array of all initialized chunk paths and their lengths.
    # Listing the array's own directory (rather than the chunk key prefix) keeps this
    # correct when that prefix isn't a whole path component, e.g. "air/c." for a V3
    # array whose chunk key separator is "." — see chunks_prefix.
    stripped_keys, full_paths, all_lengths = await build_1d_chunk_mapping(
        obs_store,
        store_base_uri,
        join_url(array_path, ""),
        join_url(array_path, on_disk_zarr_format.chunks_prefix(on_disk_separator)),
        on_disk_zarr_format,
        on_disk_separator,
        len(metadata.shape),
    )

    return chunk_entries_to_manifest(
        stripped_keys,
        full_paths,
        np.zeros(len(stripped_keys), dtype=np.uint64),
        all_lengths,
        separator=on_disk_separator,
        chunk_grid_shape=chunk_grid_shape,
    )


async def build_1d_chunk_mapping(
    obs_store: ObstoreStore,
    store_base_uri: str,
    list_prefix: str,
    array_chunks_prefix: str,
    zarr_format: ZarrFormat,
    chunk_key_separator: str,
    ndim: int,
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
    list_prefix
        Store-relative path prefix to list (e.g. "air/"). obstore matches a list prefix
        one whole path component at a time, so this must end at a "/" boundary even when
        ``array_chunks_prefix`` doesn't.
    array_chunks_prefix
        Store-relative prefix that every chunk key starts with, and which is stripped to
        get the chunk's coordinates (e.g. "air/c/", "air/c.", or "air/").
    zarr_format
        The zarr format version.
    chunk_key_separator
        The chunk key separator used on disk (e.g. ``"."`` or ``"/"``).
    ndim
        Number of dimensions of the array, i.e. how many coordinate components a
        genuine chunk key has.

    Returns
    -------
    Tuple of (stripped_keys, full_paths, sizes) as numpy arrays.
    """
    path_batches: list[np.ndarray] = []
    size_batches: list[np.ndarray] = []
    stream = obs_store.list_async(prefix=list_prefix, return_arrow=True)
    async for batch in stream:
        # Immediately convert to numpy arrays - we can still do efficient manipulations, and don't need any extra arrow dependencies.
        # Note: The .astype is only needed because .to_numpy converts to a numpy object array of python `str` objects, which is inefficient.
        # TODO: Change this if arrow -> numpy support for variable length strings ever improves, see https://github.com/zarr-developers/VirtualiZarr/issues/922#issuecomment-4051049630
        paths_np = batch.column("path").to_numpy().astype(np.dtypes.StringDType())
        sizes_np = batch.column("size").to_numpy()

        # filter out metadata and directory keys, leaving only valid chunk keys
        is_metadata = np.zeros(len(paths_np), dtype=bool)
        for suffix in zarr_format.metadata_key_names:
            is_metadata |= np.strings.endswith(paths_np, suffix)
        is_directory = np.strings.endswith(paths_np, "/")

        # Zero-byte "directory marker" objects (created by e.g. `aws s3 sync`, `s3fs`,
        # or `boto3.put_object(Key=prefix + "/")`) are keyed with a trailing slash, but
        # obstore's client-side path parsing strips that slash before the listed path
        # reaches here, so `is_directory` above can never catch them. Rather than
        # matching markers by name, keep only keys shaped like a genuine chunk key:
        # a literal descendant of the prefix with exactly one coordinate component per
        # dimension. A marker for the chunks directory itself ends up one character
        # shorter than the prefix, and a marker for a nested chunk subdirectory (only
        # possible when the separator is "/") ends up short a component, so both fail.
        is_descendant = np.strings.startswith(paths_np, array_chunks_prefix)
        relative_keys = np.strings.replace(paths_np, array_chunks_prefix, "", 1)
        n_components = np.strings.count(relative_keys, chunk_key_separator) + 1
        has_chunk_key_shape = is_descendant & (n_components == ndim)

        chunk_keys_mask = ~(is_metadata | is_directory) & has_chunk_key_shape

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
