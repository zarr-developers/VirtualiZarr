"""Parser for converting an icechunk Session into a VirtualiZarr ManifestStore.

Unlike the other parsers, this one operates on a live icechunk store rather
than parsing an archival file from object storage. It walks zarr groups via
the standard zarr-python API to enumerate arrays + their metadata, then for
each array consumes :meth:`icechunk.IcechunkStore.array_chunk_iterator` —
a columnar async generator of chunk references — and scatters the batches
into the dense numpy arrays that VirtualiZarr's
:meth:`ChunkManifest.from_arrays` expects.

The mapping is:

- IC virtual ref (any ``s3://`` / ``gs://`` / ``vcc://`` location, already
  resolved by icechunk) → VZ virtual ref with the resolved URL as path.
- IC native (managed) chunk → VZ virtual ref with path
  ``f"{native_chunks_prefix}/{chunk_id}"``. The prefix is supplied at
  parser construction time.
- IC inline chunk → VZ inline chunk (bytes carried in
  ``ChunkManifest._inlined``, path replaced by VZ's sentinel).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from zarr.api.asynchronous import open_group as open_group_async

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.parsers.zarr import metadata_as_v3
from virtualizarr.utils import determine_chunk_grid_shape

if TYPE_CHECKING:
    import icechunk
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry


# Mirrors the constants in icechunk-python/src/store.rs.
_KIND_VIRTUAL: int = 1
_KIND_NATIVE: int = 2
_KIND_INLINE: int = 3

# Default batch size for the iterator; can be overridden at parser construction.
_DEFAULT_BATCH_SIZE: int = 100_000

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync code, even when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class IcechunkParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an icechunk session.

    Parameters
    ----------
    native_chunks_prefix
        URL prefix to root icechunk's native (managed) chunk paths at.
        Native chunks are rendered as ``f"{native_chunks_prefix}/{chunk_id}"``.
        For an S3-backed repo at ``s3://my-bucket/my-repo``, this would be
        ``"s3://my-bucket/my-repo/chunks"``. A single trailing slash is tolerated.
        Virtual chunks ignore this — their URL is whatever icechunk has on file
        (with ``vcc://`` references already resolved).
    group
        Optional sub-group path within the icechunk store to use as the root.
    skip_variables
        Names of arrays in the group to exclude.
    batch_size
        Iterator batch size in chunks. Default 100,000.

    Examples
    --------
    >>> import icechunk
    >>> from obspec_utils.registry import ObjectStoreRegistry
    >>> from virtualizarr.parsers import IcechunkParser
    >>>
    >>> repo = icechunk.Repository.open(storage=...)  # doctest: +SKIP
    >>> session = repo.readonly_session(branch="main")  # doctest: +SKIP
    >>> parser = IcechunkParser(  # doctest: +SKIP
    ...     native_chunks_prefix="s3://my-bucket/my-repo/chunks",
    ... )
    >>> manifest_store = parser(session.store, registry=ObjectStoreRegistry({}))  # doctest: +SKIP
    """

    def __init__(
        self,
        native_chunks_prefix: str,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        self.native_chunks_prefix = native_chunks_prefix.rstrip("/")
        self.group = group
        self.skip_variables = skip_variables
        self.batch_size = batch_size

    def __call__(
        self,
        store: "icechunk.IcechunkStore",
        registry: "ObjectStoreRegistry",
    ) -> ManifestStore:
        """Parse an icechunk store and return a VZ ManifestStore."""
        coro = _construct_manifest_group(
            store=store,
            group=self.group,
            native_chunks_prefix=self.native_chunks_prefix,
            skip_variables=self.skip_variables,
            batch_size=self.batch_size,
        )
        manifest_group = _run_async(coro)
        return ManifestStore(registry=registry, group=manifest_group)


async def _construct_manifest_group(
    store: "icechunk.IcechunkStore",
    *,
    native_chunks_prefix: str,
    group: str | None = None,
    skip_variables: Iterable[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ManifestGroup:
    """Build a ManifestGroup from an icechunk zarr group."""
    zarr_group = await open_group_async(store=store, path=group, mode="r")

    array_keys = [key async for key in zarr_group.array_keys()]
    skip = set() if skip_variables is None else set(skip_variables)

    zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(k) for k in array_keys if k not in skip]
    )

    manifest_arrays = await asyncio.gather(
        *[
            _construct_manifest_array(arr, store, native_chunks_prefix, batch_size)
            for arr in zarr_arrays  # type: ignore[union-attr]
        ]
    )

    arrays = {a.basename: ma for a, ma in zip(zarr_arrays, manifest_arrays)}
    return ManifestGroup(arrays=arrays, attributes=dict(zarr_group.attrs))


async def _construct_manifest_array(
    zarr_array: "zarr.AsyncArray[Any]",
    store: "icechunk.IcechunkStore",
    native_chunks_prefix: str,
    batch_size: int,
) -> ManifestArray:
    """Assemble one array's ChunkManifest by scattering iterator batches."""
    metadata = metadata_as_v3(zarr_array.metadata)
    grid_shape = determine_chunk_grid_shape(
        metadata.shape, metadata.chunk_grid.chunk_shape
    )

    # Allocate dense buffers sized to the chunk grid. Scalar arrays
    # (grid_shape == ()) get a single-slot buffer that we reshape to () at the end.
    total = int(np.prod(grid_shape)) if grid_shape else 1
    paths = np.full(total, "", dtype=np.dtypes.StringDType())
    offsets = np.zeros(total, dtype=np.uint64)
    lengths = np.zeros(total, dtype=np.uint64)
    inlined: dict[tuple[int, ...], bytes] = {}

    prefix_with_slash = f"{native_chunks_prefix}/"

    async for batch in store.array_chunk_iterator(zarr_array.path, batch_size):
        b_coords, b_kinds, b_paths, b_offsets, b_lengths, b_inlined = batch
        n = b_coords.shape[0]
        if n == 0:
            continue

        # Flat row-major index for each chunk in the batch.
        if grid_shape:
            flat_idx = np.ravel_multi_index(b_coords.T, grid_shape)
        else:
            # Scalar array: only one possible chunk slot.
            flat_idx = np.zeros(n, dtype=np.intp)

        # Build the per-batch paths column with the right values:
        # - virtual: URL as-is
        # - native:  prepend prefix to chunk_id
        # - inline:  VZ sentinel
        path_col = np.array(b_paths, dtype=np.dtypes.StringDType())
        is_native = b_kinds == _KIND_NATIVE
        if is_native.any():
            path_col[is_native] = np.strings.add(
                prefix_with_slash, path_col[is_native]
            )
        is_inline = b_kinds == _KIND_INLINE
        if is_inline.any():
            path_col[is_inline] = INLINED_CHUNK_PATH

        # Scatter into the dense buffers.
        paths[flat_idx] = path_col
        offsets[flat_idx] = b_offsets
        lengths[flat_idx] = b_lengths

        # Inline bytes are keyed by absolute chunk coordinate, not flat index.
        for batch_i, data in b_inlined.items():
            coord = tuple(int(x) for x in b_coords[batch_i])
            inlined[coord] = data

    chunk_manifest = ChunkManifest.from_arrays(
        paths=paths.reshape(grid_shape or (1,)),
        offsets=offsets.reshape(grid_shape or (1,)),
        lengths=lengths.reshape(grid_shape or (1,)),
        validate_paths=False,
        inlined=inlined or None,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
