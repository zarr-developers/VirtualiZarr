"""Translate xarray's reindex/alignment indexer into a chunk-grid map."""

from __future__ import annotations

import numpy as np

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
