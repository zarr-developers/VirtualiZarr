from types import EllipsisType
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from virtualizarr.manifests.array_api import expand_dims

# indexer without new axes
T_BasicIndexer_1d: TypeAlias = int | slice | np.ndarray
# indexer without ellipses
T_SimpleIndexer_1d: TypeAlias = T_BasicIndexer_1d | None
T_SimpleIndexer: TypeAlias = tuple[T_SimpleIndexer_1d, ...]
# general valid indexer representing any possible user input
T_Indexer_1d: TypeAlias = T_SimpleIndexer_1d | EllipsisType
T_IndexerTuple: TypeAlias = tuple[T_Indexer_1d, ...]
T_Indexer: TypeAlias = T_Indexer_1d | tuple[T_Indexer_1d, ...]


if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray


def index(marr: "ManifestArray", indexer: T_Indexer) -> "ManifestArray":
    """Index into a ManifestArray"""
    indexer_tuple = check_and_sanitize_indexer_type(indexer)
    indexer_without_ellipsis = check_shape_and_maybe_replace_ellipsis(
        indexer_tuple, marr.ndim
    )
    return apply_indexer(marr, indexer_without_ellipsis)


def check_and_sanitize_indexer_type(key: T_Indexer) -> T_IndexerTuple:
    """Check for invalid input types, and narrow the return type to a tuple of valid 1D indexers."""
    if isinstance(key, (int, slice, EllipsisType, np.ndarray)) or key is None:
        indexer = (key,)
    elif isinstance(key, tuple):
        for dim_indexer in key:
            if (
                not isinstance(dim_indexer, (int, slice, EllipsisType, np.ndarray))
                and dim_indexer is not None
            ):
                raise TypeError(
                    f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
                )
        indexer = key
    else:
        raise TypeError(
            f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
        )
    return indexer


def check_shape_and_maybe_replace_ellipsis(
    indexer: T_IndexerTuple, arr_ndim: int
) -> T_SimpleIndexer:
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
            return indexer


def replace_single_ellipsis(
    indexer: T_Indexer_1d, arr_ndim: int, num_single_axis_indexing_expressions: int
) -> T_SimpleIndexer_1d:
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

    return tuple(indexer_as_list)


def apply_indexer(marr: "ManifestArray", indexer: T_SimpleIndexer) -> "ManifestArray":
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
    """Actually applies indexes to subset along each dimension."""

    # at this point there should be no ellipsis, no Nones, and one 1D indexer for each axis.
    assert len(indexer_without_newaxes) == marr.ndim

    output_arr = marr
    for ind, axis_indexer in enumerate(indexer_without_newaxes):
        if isinstance(axis_indexer, slice):
            if slice_is_no_op(axis_indexer, axis_length=marr.shape[ind]):
                pass
            else:
                NotImplementedError(
                    f"Unsupported indexer. Indexing within a ManifestArray using ints or slices is not yet supported (see GitHub issue #51), but received {axis_indexer}"
                )
        elif isinstance(axis_indexer, int):
            # TODO cover possibility of indexing into a length-1 dimension (which just removes that dimension)?
            raise NotImplementedError(
                f"Unsupported indexer. Indexing within a ManifestArray using ints or slices is not yet supported (see GitHub issue #51), but received {axis_indexer}"
            )
        elif isinstance(axis_indexer, np.ndarray):
            raise NotImplementedError(
                f"Unsupported indexer. So-called 'fancy indexing' via numpy arrays is not supported, but received {axis_indexer}"
            )
        else:
            # should never get here
            raise TypeError(f"Invalid indexer type: {axis_indexer}")

    return output_arr


def slice_is_no_op(slice_indexer_1d: slice, axis_length: int) -> bool:
    if slice_indexer_1d == slice(None):
        return True
    elif slice_indexer_1d == slice(0, axis_length, 1):
        return True
    else:
        return False
