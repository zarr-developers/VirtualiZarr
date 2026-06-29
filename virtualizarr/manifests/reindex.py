from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_SPLIT_MSG = (
    "Cannot reindex lazily: the requested target labels would require splitting "
    "or sub-chunk reordering of a source chunk, which VirtualiZarr does not do "
    "(it would require reading chunk bytes). Only whole-chunk appends, inserts, "
    "and reorders are supported. See https://github.com/zarr-developers/VirtualiZarr/issues/51."
)


def chunk_index_map(
    source_labels: Any,
    target_labels: Any,
    chunk_size: int,
) -> list[int | None]:
    """
    Translate a label-space reindex into a chunk-grid map.

    Returns one entry per target chunk slot: the index of the source chunk to
    copy into that slot, or ``None`` for an all-missing (null-path) chunk that
    reads back as the array's ``fill_value``.

    Raises
    ------
    NotImplementedError
        If the reindex cannot be expressed at chunk granularity (would split,
        partially cover, or sub-chunk-reorder a source chunk).
    """
    source_index = pd.Index(source_labels)
    target_index = pd.Index(target_labels)

    n_source = len(source_index)
    # -1 where a target label is absent from the source. Raises if source non-unique.
    pos = source_index.get_indexer(target_index)

    n_target = len(target_index)
    chunk_map: list[int | None] = []
    for start in range(0, n_target, chunk_size):
        block = pos[start : start + chunk_size]

        if np.all(block == -1):
            chunk_map.append(None)
            continue
        if np.any(block == -1):
            # block mixes present and missing labels -> would split a chunk
            raise NotImplementedError(_SPLIT_MSG)

        s = int(block[0])
        # must be a contiguous, ascending, step-1 run (no sub-chunk reorder)
        if not np.array_equal(block, np.arange(s, s + len(block))):
            raise NotImplementedError(_SPLIT_MSG)
        # must start on a source chunk boundary
        if s % chunk_size != 0:
            raise NotImplementedError(_SPLIT_MSG)
        src_chunk = s // chunk_size
        # block length must match that source chunk's actual size (handles trailing partial)
        src_chunk_size = min(chunk_size, n_source - src_chunk * chunk_size)
        if len(block) != src_chunk_size:
            raise NotImplementedError(_SPLIT_MSG)

        chunk_map.append(src_chunk)

    return chunk_map
