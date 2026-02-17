"""
Vendored utilities from zarr-python for chunk grid handling.

See https://github.com/zarr-developers/zarr-python/pull/3534
"""

from typing import Any

from zarr.core.chunk_grids import ChunkGrid


def _is_nested_sequence(chunks: Any) -> bool:
    """
    Check if chunks is a nested sequence (tuple of tuples/lists).

    Returns True for inputs like [[10, 20], [5, 5]] or [(10, 20), (5, 5)].
    Returns False for flat sequences like (10, 10) or [10, 10].

    Vendored from https://github.com/zarr-developers/zarr-python/pull/3534
    """
    if isinstance(chunks, str | int | ChunkGrid):
        return False

    if not hasattr(chunks, "__iter__"):
        return False

    try:
        first_elem = next(iter(chunks), None)
        if first_elem is None:
            return False
        return hasattr(first_elem, "__iter__") and not isinstance(
            first_elem, str | bytes | int
        )
    except (TypeError, StopIteration):
        return False
