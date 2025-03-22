from types import NoneType, EllipsisType
from typing import Union


def array_slice_to_chunk_slice(
    array_slice: slice,
    arr_length: int,
    chunk_length: int,
) -> slice:
    """
    Translate a slice into an array into a corresponding slice into the underlying chunk grid.

    Will raise on any array slices that require slicing within individual chunks.
    """

    if chunk_length == 1:
        # alot of indexing is possible only in this case, because this is basically just a normal array along that axis
        chunk_slice = array_slice
        return chunk_slice

    # Check that start of slice aligns with start of a chunk
    if array_slice.start % chunk_length != 0:
        raise NotImplementedError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    # Check that slice spans integer number of chunks
    slice_length = array_slice.stop - array_slice.start
    if slice_length % chunk_length != 0:
        raise NotImplementedError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    index_of_first_chunk = int(array_slice.start / chunk_length)
    n_chunks = int(slice_length / chunk_length)

    chunk_slice = slice(index_of_first_chunk, index_of_first_chunk + n_chunks, 1)

    return chunk_slice


def possibly_expand_trailing_ellipses(indexer, ndim: int) -> tuple[Union[int, slice, EllipsisType, None], ...]:
    if indexer[-1] == ...:
        extra_slices_needed = ndim - (len(indexer) - 1)
        *indexer_1d, ellipsis = indexer
        return tuple(tuple(indexer_1d) + (slice(None),) * extra_slices_needed)
    else:
        return indexer


def array_indexer_to_chunk_grid_indexer(
    indexer_nd,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    """Convert an indexer in array element space into the corresponding indexer in chunk grid space."""

    chunk_slices = []
    new_arr_shape = []
    for axis_num, (indexer_1d, arr_length, chunk_length) in enumerate(
        zip(indexer_nd, shape, chunks)
    ):
        if isinstance(indexer_1d, int):
            array_slice_1d = slice(indexer_1d, indexer_1d + 1, 1)
        elif isinstance(indexer_1d, NoneType):
            array_slice_1d = slice(0, arr_length, 1)
        elif isinstance(indexer_1d, slice):
            array_slice_1d = slice(
                indexer_1d.start if indexer_1d.start is not None else 0,
                indexer_1d.stop if indexer_1d.stop is not None else arr_length,
                indexer_1d.step if indexer_1d.step is not None else 1,
            )
        else:
            # TODO we could attempt to also support indexing with numpy arrays
            raise TypeError(
                f"Can only perform indexing with keys of type (int, slice, EllipsisType, NoneType), but got type {type(indexer_1d)} for axis {axis_num}"
            )

        chunk_slice_1d = array_slice_to_chunk_slice(
            array_slice_1d, arr_length, chunk_length
        )
        chunk_slices.append(chunk_slice_1d)

        n_elements_in_slice = abs(
            (array_slice_1d.start - array_slice_1d.stop) / array_slice_1d.step
        )
        new_arr_shape.append(n_elements_in_slice)

        return tuple(chunk_slices), tuple(new_arr_shape)
