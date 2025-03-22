from types import NoneType, EllipsisType
from typing import Union

from virtualizarr.utils import determine_chunk_grid_shape

from zarr.core.indexing import BasicIndexer, IntDimIndexer, SliceDimIndexer



# TODO write a custom message for this
class SubChunkIndexingError(IndexError):
    ...


def array_indexer_to_chunk_grid_indexer(
    indexer: BasicIndexer,
) -> BasicIndexer:
    """
    Translate an indexer into an array into a corresponding indexer into the underlying chunk grid.

    As compressed chunks cannot be subdivided, this will raise on any array slices that require slicing within individual chunks.
    """

    print(f"{indexer=}")

    [
        print(chunk_coords, chunk_selection, out_selection, is_complete_chunk)
        for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
    ]

    # TODO move this check outside? Because we can arbitrarily index into uncompressed arrays...
    if not all(is_complete_chunk for _, _, _, is_complete_chunk in indexer):
        raise SubChunkIndexingError()
    
    array_shape = indexer.shape
    chunk_shape = indexer.chunk_grid.chunk_shape
    chunk_grid_shape = determine_chunk_grid_shape(array_shape, chunk_shape)

    chunk_grid_dim_indexers: list[IntDimIndexer | SliceDimIndexer] = []
    for dim_indexer, dim_len, dim_chunk_len in zip(
        indexer.dim_indexers,
        indexer.shape,
        chunk_grid_shape,
        strict=True,
    ):
        chunk_grid_dim_indexer: IntDimIndexer | SliceDimIndexer
        if isinstance(dim_indexer, IntDimIndexer):
            if dim_len == 1:
                chunk_grid_dim_indexer = dim_indexer
            else:
                raise SubChunkIndexingError
        
        elif isinstance(dim_indexer, SliceDimIndexer):
            dim_indexer = array_slice_to_chunk_grid_slice(dim_indexer)

        chunk_grid_dim_indexers.append(chunk_grid_dim_indexer)

    obj = BasicIndexer.__new__()
    obj.dim_indexers = chunk_grid_dim_indexers
    obj.shape = tuple(s.nitems for s in chunk_grid_dim_indexers if not isinstance(s, IntDimIndexer))
    obj.drop_axes = ()

    return obj


def array_slice_to_chunk_grid_slice(
    array_slice: SliceDimIndexer,
) -> SliceDimIndexer:
    """
    Translate a slice into an array into a corresponding slice into the underlying chunk grid.

    Will raise on any array slices that would require slicing within individual chunks.
    """

    arr_length = array_slice.dim_len
    chunk_length = array_slice.dim_chunk_len

    if chunk_length == 1:
        # alot of indexing is possible only in this case, because this is basically just a normal array along that axis
        chunk_slice = array_slice
        return chunk_slice

    # Check that start of slice aligns with start of a chunk
    if array_slice.start % chunk_length != 0:
        raise IndexError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    # Check that slice spans integer number of chunks
    slice_length = array_slice.stop - array_slice.start
    if slice_length % chunk_length != 0:
        raise IndexError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    index_of_first_chunk = int(array_slice.start / chunk_length)
    n_chunks = int(slice_length / chunk_length)

    chunk_slice = slice(index_of_first_chunk, index_of_first_chunk + n_chunks, 1)

    return SliceDimIndexer(
        dim_sel=chunk_slice, 
        # TODO which dim does this refer to? That of the chunk grid?
        dim_len=...,
        dim_chunk_len=...,
    )
