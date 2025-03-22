from zarr.core.indexing import (
    BasicIndexer,
    BasicSelection,
    BasicSelector,
    IntDimIndexer,
    SliceDimIndexer,
)


# TODO write a custom message for this
class SubChunkIndexingError(IndexError): ...


def array_indexer_to_chunk_grid_indexer(
    indexer: BasicIndexer,
    arr_shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
) -> BasicIndexer:
    """
    Translate an indexer into an array into a corresponding indexer into the underlying chunk grid.

    As compressed chunks cannot be subdivided, this will raise on any array slices that require slicing within individual chunks.

    Parameters
    ----------
    indexer
    arr_shape : tuple[int, ...]
        Shape of the ManifestArray we are trying to index into.
    """

    print(f"{indexer=}")

    [
        print(chunk_coords, chunk_selection, out_selection, is_complete_chunk)
        for chunk_coords, chunk_selection, out_selection, is_complete_chunk in indexer
    ]

    # TODO move this check outside? Because we can arbitrarily index into uncompressed arrays...
    # TODO or is that unrelated? Uncompressed arrays allow us to choose arbitrary chunking at reference generation time
    if not all(is_complete_chunk for _, _, _, is_complete_chunk in indexer):
        raise SubChunkIndexingError()

    if indexer.drop_axes:
        raise NotImplementedError

    # TODO does the array shape have to match the indexer shape? Should this have been checked already?
    chunk_grid_dim_indexers: list[IntDimIndexer | SliceDimIndexer] = []
    for (
        dim_indexer,
        dim_len,
        dim_chunk_len,
    ) in zip(
        indexer.dim_indexers,
        arr_shape,
        chunk_shape,
        # chunk_grid_shape,  # TODO do we really not need this??
        strict=True,
    ):
        chunk_grid_dim_indexer: IntDimIndexer | SliceDimIndexer
        if isinstance(dim_indexer, IntDimIndexer):
            chunk_grid_dim_indexer = array_int_indexer_to_chunk_grid_int_indexer(dim_indexer)

        elif isinstance(dim_indexer, SliceDimIndexer):
            chunk_grid_dim_indexer = array_slice_indexer_to_chunk_grid_slice_indexer(dim_indexer)

        chunk_grid_dim_indexers.append(chunk_grid_dim_indexer)

    # TODO check this - I'm not sure if I've understood the "shape" of an indexer correctly
    chunk_grid_dim_indexer_shape = tuple(
        s.nitems for s in chunk_grid_dim_indexers if not isinstance(s, IntDimIndexer)
    )

    # The BasicIndexer constructor doesn't allow us to set the attributes we want to
    # Also BasicIndexer is a frozen dataclass so we have to use .__setattr__()
    chunk_grid_indexer = object.__new__(BasicIndexer)
    object.__setattr__(chunk_grid_indexer, "dim_indexers", chunk_grid_dim_indexers)
    object.__setattr__(chunk_grid_indexer, "shape", chunk_grid_dim_indexer_shape)
    object.__setattr__(chunk_grid_indexer, "drop_axes", ())

    return chunk_grid_indexer


def array_int_indexer_to_chunk_grid_int_indexer(
    array_int_indexer: IntDimIndexer,
) -> IntDimIndexer:
    """
    Translate an integer indexer into an array into a corresponding integer indexer into the underlying chunk grid.

    Will raise on any array indexer that would require slicing within individual chunks.
    """
    if array_int_indexer.dim_chunk_len == 1:
        # degenerate case where array space == chunk grid space, so array indexer == chunk grid indexer
        # TODO pull out the degenerate case for both ints and slices?
        chunk_grid_dim_indexer = array_int_indexer
    else:
        # TODO what about case of array of integers that don't split up chunks?
        raise SubChunkIndexingError
    
    return chunk_grid_dim_indexer


def array_slice_indexer_to_chunk_grid_slice_indexer(
    arr_slice_dim_indexer: SliceDimIndexer,
) -> SliceDimIndexer:
    """
    Translate a slice into an array into a corresponding slice into the underlying chunk grid.

    Will raise on any array slices that would require slicing within individual chunks.
    """

    arr_length = arr_slice_dim_indexer.dim_len
    chunk_length = arr_slice_dim_indexer.dim_chunk_len

    if chunk_length == 1:
        # degenerate case where array space == chunk grid space, so array indexer == chunk grid indexer
        chunk_slice = slice(
            arr_slice_dim_indexer.start,
            arr_slice_dim_indexer.stop,
            arr_slice_dim_indexer.step,
        )

    # Check that start of slice aligns with start of a chunk
    if arr_slice_dim_indexer.start % chunk_length != 0:
        raise SubChunkIndexingError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    # Check that slice spans integer number of chunks
    slice_length = arr_slice_dim_indexer.stop - arr_slice_dim_indexer.start
    if slice_length % chunk_length != 0:
        raise SubChunkIndexingError(
            f"Cannot index ManifestArray axis of length {arr_length} and chunk length {chunk_length} with {array_slice} as slice would split individual chunks"
        )

    index_of_first_chunk = int(arr_slice_dim_indexer.start / chunk_length)
    n_chunks = int(slice_length / chunk_length)

    chunk_slice = slice(index_of_first_chunk, index_of_first_chunk + n_chunks, 1)

    chunk_grid_slice_dim_indexer = SliceDimIndexer(
        dim_sel=chunk_slice,
        # TODO which dim does this refer to? That of the chunk grid?
        dim_len=...,
        dim_chunk_len=...,
    )

    return chunk_grid_slice_dim_indexer


def indexer_to_selection(indexer: BasicIndexer) -> BasicSelection:
    """Translate an indexer into a selection that numpy can understand."""
    selection = []
    for dim_indexer in indexer.dim_indexers:
        dim_selection: BasicSelector
        if isinstance(dim_indexer, IntDimIndexer):
            # avoid numpy returning a scalar value instead of np.ndarray
            dim_selection = slice(
                dim_indexer.dim_sel,
                dim_indexer.dim_sel + 1,
            )
        elif isinstance(dim_indexer, SliceDimIndexer):
            dim_selection = slice(
                dim_indexer.start,
                dim_indexer.stop,
                dim_indexer.step,
            )
        else:
            raise NotImplementedError

        selection.append(dim_selection)

    return tuple(selection)
