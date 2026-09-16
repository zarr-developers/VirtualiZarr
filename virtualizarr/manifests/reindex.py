"""Reindex a ManifestArray: indexer -> chunk-grid map -> remapped manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH, ChunkManifest
from virtualizarr.manifests.utils import copy_and_replace_metadata

if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray

_SPLIT_MSG = (
    "Cannot reindex/align lazily: the requested labels would require splitting "
    "or sub-chunk reordering of a source chunk, which VirtualiZarr does not do "
    "(it would require reading chunk bytes). Only whole-chunk appends, inserts, "
    "and reorders are supported. See https://github.com/zarr-developers/VirtualiZarr/issues/51."
)


def chunk_map_from_indexer(
    indexer: np.ndarray,
    chunk_size: int,
    source_len: int,
) -> list[int | None]:
    """
    Partition a reindex/alignment indexer into a chunk-grid map.

    xarray's reindex hands the backing array an integer indexer along the
    reindexed axis, with ``-1`` marking positions absent from the source (the
    would-be-fill positions). This partitions that indexer by chunk and returns,
    for each target chunk slot, either the index of the source chunk to copy into
    it, or ``None`` for an all-missing (null-path) chunk that reads back as
    ``fill_value``.

    Each chunk-sized block of the indexer must be either entirely ``-1`` (→ null
    chunk) or a contiguous, ascending, chunk-aligned run of source positions
    (→ that source chunk). Anything else — a block mixing present and missing
    positions, a sub-chunk reorder, or an unaligned start — raises, because it
    cannot be expressed without splitting a chunk.

    Parameters
    ----------
    indexer
        Integer array (xarray's positional indexer), ``-1`` where missing.
    chunk_size
        Chunk size along this axis (number of elements per chunk).
    source_len
        Length of the source axis in elements (to size the trailing partial chunk).

    Returns
    -------
    list of (int or None)
        One entry per target chunk slot.

    Raises
    ------
    NotImplementedError
        If the indexer cannot be expressed at chunk granularity.
    """
    pos = np.asarray(indexer)
    chunk_map: list[int | None] = []
    for start in range(0, len(pos), chunk_size):
        block = pos[start : start + chunk_size]

        if np.all(block == -1):
            chunk_map.append(None)
            continue
        if np.any(block == -1):
            raise NotImplementedError(_SPLIT_MSG)

        s = int(block[0])
        if not np.array_equal(block, np.arange(s, s + len(block))):
            raise NotImplementedError(_SPLIT_MSG)
        if s % chunk_size != 0:
            raise NotImplementedError(_SPLIT_MSG)
        src_chunk = s // chunk_size
        src_chunk_size = min(chunk_size, source_len - src_chunk * chunk_size)
        if len(block) != src_chunk_size:
            raise NotImplementedError(_SPLIT_MSG)

        chunk_map.append(src_chunk)

    return chunk_map


def reindex_axis(
    marr: "ManifestArray", axis: int, chunk_map: list[int | None], new_size: int
) -> "ManifestArray":
    """
    Return a new ManifestArray with the chunk grid along ``axis`` remapped.

    Each entry of ``chunk_map`` is the source chunk index to copy into that
    target chunk slot, or ``None`` for a missing (null-path) chunk that reads
    back as ``fill_value``. ``new_size`` is the new length of ``axis`` in
    elements; the chunk size is unchanged.
    """
    # deferred to break the array -> indexing -> reindex import cycle
    from virtualizarr.manifests.array import ManifestArray

    manifest = marr.manifest
    src_paths = manifest._paths
    src_offsets = manifest._offsets
    src_lengths = manifest._lengths

    new_grid_shape = list(src_paths.shape)
    new_grid_shape[axis] = len(chunk_map)

    new_paths = np.full(
        new_grid_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType()
    )
    new_offsets = np.zeros(new_grid_shape, dtype=np.uint64)
    new_lengths = np.zeros(new_grid_shape, dtype=np.uint64)

    new_inlined: dict[tuple[int, ...], bytes] = {}
    for new_idx, src_chunk in enumerate(chunk_map):
        if src_chunk is None:
            continue  # leave this slab as missing/null
        src_slice: list[Any] = [slice(None)] * src_paths.ndim
        src_slice[axis] = src_chunk
        dst_slice: list[Any] = [slice(None)] * src_paths.ndim
        dst_slice[axis] = new_idx
        new_paths[tuple(dst_slice)] = src_paths[tuple(src_slice)]
        new_offsets[tuple(dst_slice)] = src_offsets[tuple(src_slice)]
        new_lengths[tuple(dst_slice)] = src_lengths[tuple(src_slice)]
        # re-key any inlined chunks that lived in this source slab
        for key, data in manifest._inlined.items():
            if key[axis] == src_chunk:
                shifted = list(key)
                shifted[axis] = new_idx
                new_inlined[tuple(shifted)] = data

    new_manifest = ChunkManifest.from_arrays(
        paths=new_paths,
        offsets=new_offsets,
        lengths=new_lengths,
        validate_paths=False,
        inlined=new_inlined if new_inlined else None,
    )

    new_shape = list(marr.shape)
    new_shape[axis] = new_size
    new_metadata = copy_and_replace_metadata(
        old_metadata=marr.metadata, new_shape=new_shape
    )

    return ManifestArray(chunkmanifest=new_manifest, metadata=new_metadata)
