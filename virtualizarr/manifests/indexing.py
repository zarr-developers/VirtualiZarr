from types import EllipsisType
from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np

from virtualizarr.manifests.array_api import expand_dims
from virtualizarr.manifests.manifest import ChunkManifest
from virtualizarr.manifests.utils import copy_and_replace_metadata

# indexer with only basic selectors, no new axes or ellipsis
T_BasicIndexer_1d: TypeAlias = int | slice | np.ndarray
# indexer allowing new axes but without ellipses
T_SimpleIndexer_1d: TypeAlias = T_BasicIndexer_1d | None
# general valid indexer representing any possible user input
T_Indexer_1d: TypeAlias = T_SimpleIndexer_1d | EllipsisType
T_Indexer: TypeAlias = T_Indexer_1d | tuple[T_Indexer_1d, ...]


class SubChunkIndexingError(ValueError):
    """
    Raised when an indexer would split individual chunks of a compressed ManifestArray.
    """


if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray


def index(marr: "ManifestArray", indexer: T_Indexer) -> "ManifestArray":
    """Index into a ManifestArray"""
    indexer_tuple = check_and_sanitize_indexer_type(indexer)
    indexer_without_ellipsis = check_shape_and_maybe_replace_ellipsis(
        indexer_tuple, marr.ndim
    )
    return apply_indexer(marr, indexer_without_ellipsis)


def check_and_sanitize_indexer_type(key: T_Indexer) -> tuple[T_Indexer_1d, ...]:
    """Check for invalid input types, and narrow the return type to a tuple of valid 1D indexers."""
    if isinstance(key, (int, slice, EllipsisType, np.ndarray)) or key is None:
        indexer = cast(tuple[T_Indexer_1d, ...], (key,))
    elif isinstance(key, tuple):
        for dim_indexer in key:
            if (
                not isinstance(dim_indexer, (int, slice, np.ndarray))
                and dim_indexer is not None
                and dim_indexer is not ...
            ):
                raise TypeError(
                    f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
                )
        indexer = cast(tuple[T_Indexer_1d, ...], key)
    else:
        raise TypeError(
            f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
        )
    return indexer


def check_shape_and_maybe_replace_ellipsis(
    indexer: tuple[T_Indexer_1d, ...], arr_ndim: int
) -> tuple[T_SimpleIndexer_1d, ...]:
    """Deal with any ellipses, potentially by expanding the indexer to match the shape of the array."""

    num_single_axis_indexing_expressions = len(
        [ind_1d for ind_1d in indexer if ind_1d is not None and ind_1d is not ...]
    )
    num_ellipses = len([ind_1d for ind_1d in indexer if ind_1d is ...])

    if num_ellipses > 1:
        raise ValueError(
            f"Invalid indexer. Indexers containing multiple Ellipses are invalid, but indexer={indexer} contains {num_ellipses} ellipses"
        )

    bad_shape_error_msg = (
        "Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions less than or equal to the length of the array. "
        f"However indexer={indexer} has {num_single_axis_indexing_expressions} single-axis indexing expressions and array has {arr_ndim} dimensions."
        "\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
        "or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
    )

    if num_ellipses == 1:
        if num_single_axis_indexing_expressions > arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return replace_single_ellipsis(
                indexer, arr_ndim, num_single_axis_indexing_expressions
            )
    else:  # num_ellipses == 0
        if num_single_axis_indexing_expressions != arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return cast(tuple[T_SimpleIndexer_1d, ...], indexer)


def replace_single_ellipsis(
    indexer: tuple[T_Indexer_1d, ...],
    arr_ndim: int,
    num_single_axis_indexing_expressions: int,
) -> tuple[T_SimpleIndexer_1d, ...]:
    """
    Replace ellipsis with 0 or more slice(None)s until there are ndim single-axis indexing expressions (so ignoring Nones).
    """
    indexer_as_list = list(indexer)

    # find this position before modifying in-place
    position_of_ellipsis = indexer_as_list.index(...)

    # need to remove the ellipsis separate from replacement, because the ellipsis may have been totally superfluous already,
    # and it still needs to be removed even if no further replacement will occur
    indexer_as_list.remove(...)

    # replace ellipsis with the equivalent number of no-op slices
    num_extra_slices_needed = arr_ndim - num_single_axis_indexing_expressions
    new_slices = [slice(None)] * num_extra_slices_needed

    # insert multiple elements into one position
    indexer_as_list[position_of_ellipsis:position_of_ellipsis] = new_slices

    return cast(tuple[T_SimpleIndexer_1d, ...], tuple(indexer_as_list))


def apply_indexer(
    marr: "ManifestArray", indexer: tuple[T_SimpleIndexer_1d, ...]
) -> "ManifestArray":
    """
    Apply the simplified indexer to all dimensions of the array.

    Iterates over the axes and applies each indexer one-by-one.
    Encountering a None means inserting a new axis at that position.
    """
    # handles composition of subsetting and expanding dims by following the two-step approach described in https://github.com/data-apis/array-api/pull/408#issuecomment-1091056873

    indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...] = tuple(
        [ind_1d for ind_1d in indexer if ind_1d is not None]
    )
    output_arr = apply_selection(marr, indexer_without_newaxes)

    for position, axis_indexer in enumerate(indexer):
        if axis_indexer is None:
            output_arr = expand_dims(output_arr, axis=position)

    return output_arr


def apply_selection(
    marr: "ManifestArray", indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...]
) -> "ManifestArray":
    """
    Apply chunk-aligned subsetting along each dimension.

    Slices and integer indexers are only supported if they align with chunk boundaries, since
    splitting individual chunks would require loading their bytes. See GitHub issue #51.
    """
    from virtualizarr.manifests.array import ManifestArray

    # at this point there should be no ellipsis, no Nones, and one 1D indexer for each axis.
    assert len(indexer_without_newaxes) == marr.ndim

    # validate types and reject anything we can never support per-axis
    for indexer_1d in indexer_without_newaxes:
        if isinstance(indexer_1d, np.ndarray):
            raise NotImplementedError(
                f"Unsupported indexer. So-called 'fancy indexing' via numpy arrays is not supported, but received {indexer_1d}"
            )
        if not isinstance(indexer_1d, (int, slice)):
            raise TypeError(f"Invalid indexer type: {indexer_1d}")

    new_shape: list[int] = []
    chunk_grid_slices: list[slice] = []
    for axis_length, chunk_size, indexer_1d in zip(
        marr.shape, marr.chunks, indexer_without_newaxes, strict=True
    ):
        chunk_grid_slice, new_axis_length = _compute_chunk_aligned_selection_1d(
            indexer_1d, axis_length=axis_length, chunk_size=chunk_size
        )
        chunk_grid_slices.append(chunk_grid_slice)
        new_shape.append(new_axis_length)

    chunk_grid_slices_tuple = tuple(chunk_grid_slices)

    # short-circuit if every axis selects the whole chunk grid (a no-op)
    if all(
        cgs == slice(0, dim, 1)
        for cgs, dim in zip(chunk_grid_slices_tuple, marr.manifest.shape_chunk_grid)
    ):
        return marr

    new_manifest = _subset_manifest(marr.manifest, chunk_grid_slices_tuple)
    new_metadata = copy_and_replace_metadata(marr.metadata, new_shape=new_shape)
    return ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)


def _compute_chunk_aligned_selection_1d(
    indexer_1d: int | slice, axis_length: int, chunk_size: int
) -> tuple[slice, int]:
    """
    Translate a 1D array-space indexer (int or slice) into a chunk-grid slice plus the new axis length.

    Raises SubChunkIndexingError if the selection would require splitting individual chunks.
    """
    if isinstance(indexer_1d, int):
        # Integer indexing is treated as a length-1 slice; we don't drop dimensions because that
        # would require loading the underlying data for the array-API conformance reasons that
        # ManifestArray operates on chunk references rather than values.
        i = indexer_1d
        if i < 0:
            i += axis_length
        if not (0 <= i < axis_length):
            raise IndexError(
                f"index {indexer_1d} is out of bounds for axis with size {axis_length}"
            )
        start, stop, step = i, i + 1, 1
    else:
        start, stop, step = indexer_1d.indices(axis_length)

    if step != 1:
        raise SubChunkIndexingError(
            f"step != 1 is not supported for chunk-aligned indexing, got step={step}"
        )

    if start % chunk_size != 0:
        raise SubChunkIndexingError(
            f"Cannot index ManifestArray axis of length {axis_length} and chunk length "
            f"{chunk_size} with {indexer_1d!r}: slice would split individual chunks, "
            "which a ManifestArray cannot do without loading the underlying data."
        )

    # The final chunk may legitimately be partial, so allow stop == axis_length even if
    # axis_length % chunk_size != 0; otherwise stop must land on a chunk boundary.
    if stop != axis_length and stop % chunk_size != 0:
        raise SubChunkIndexingError(
            f"Cannot index ManifestArray axis of length {axis_length} and chunk length "
            f"{chunk_size} with {indexer_1d!r}: slice would split individual chunks, "
            "which a ManifestArray cannot do without loading the underlying data."
        )

    chunk_start = start // chunk_size
    # ceil-divide so that a partial final chunk is included when stop == axis_length
    chunk_stop = -(-stop // chunk_size)
    return slice(chunk_start, chunk_stop, 1), stop - start


def _subset_manifest(
    manifest: ChunkManifest, chunk_grid_slices: tuple[slice, ...]
) -> ChunkManifest:
    """Subset a ChunkManifest by slicing its underlying chunk-grid arrays."""
    new_paths = manifest._paths[chunk_grid_slices]
    new_offsets = manifest._offsets[chunk_grid_slices]
    new_lengths = manifest._lengths[chunk_grid_slices]

    if manifest._inlined:
        starts = tuple(s.start for s in chunk_grid_slices)
        stops = tuple(s.stop for s in chunk_grid_slices)
        new_inlined = {
            tuple(idx - start for idx, start in zip(coords, starts)): data
            for coords, data in manifest._inlined.items()
            if all(start <= idx < stop for idx, start, stop in zip(coords, starts, stops))
        }
    else:
        new_inlined = {}

    return ChunkManifest.from_arrays(
        paths=new_paths,
        offsets=new_offsets,
        lengths=new_lengths,
        inlined=new_inlined,
        validate_paths=False,
    )
