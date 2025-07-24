from types import EllipsisType
from typing import TypeAlias

from xarray.core.indexing import BasicIndexer
import numpy as np

from virtualizarr.manifests.array import ManifestArray
from virtualizarr.manifests.array_api import expand_dims


T_SimpleIndexer_1d: TypeAlias = int | slice | None | np.ndarray
T_SimpleIndexer: TypeAlias = tuple[T_SimpleIndexer_1d, ...]
T_Indexer_1d: TypeAlias = T_SimpleIndexer_1d | EllipsisType
T_IndexerTuple: TypeAlias = tuple[T_Indexer_1d, ...]
T_ValidIndexer: TypeAlias = T_Indexer_1d | tuple[T_Indexer_1d, ...]


def index(marr: ManifestArray, indexer: T_Indexer) -> ManifestArray:
    """Index into a ManifestArray"""
    indexer_tuple = check_and_sanitize_indexer_type(indexer)
    indexer_without_ellipsis = check_shape_and_replace_ellipsis(indexer_tuple, marr.ndim)
    return apply_indexer(marr, indexer_without_ellipsis)


def check_and_sanitize_indexer_type(key: T_Indexer) -> T_IndexerTuple:
    """Check for invalid input types, and narrow the return type to a tuple of valid 1D indexers."""
    if isinstance(key, BasicIndexer):
        indexer = key.tuple
    elif isinstance(key, (int, slice, EllipsisType, np.ndarray)) or key is None:
        if isinstance(key, np.ndarray):
            raise NotImplementedError(
                f"indexing with so-called 'fancy indexing' via numpy arrays is not supported. Got {key}"
            )
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

            if isinstance(key, np.ndarray):
                raise NotImplementedError(
                    f"indexing with so-called 'fancy indexing' via numpy arrays is not supported. Got {key}"
                )
        indexer = key
    else:
        raise TypeError(
            f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
        )
    return indexer


def check_shape_and_replace_ellipsis(indexer: T_Indexer, arr_ndim: int) -> T_SimpleIndexer:
    """Deal with any ellipses, potentially by expanding the indexer to match the shape of the array."""
    num_single_axis_indexing_expressions = len(ind_1d for ind in indexer if ind_1d != None and ind_1d != ...)
    num_ellipses = indexer.count(...)

    if num_ellipses > 1:
        raise ValueError(f"Invalid indexer. Indexers containing multiple Ellipses are invalid, but indexer={indexer} contains {num_ellipses} ellipses")
    elif num_ellipses == 1:
        # TODO expand ellipses
        if num_single_axis_indexing_expressions > arr_ndim:
            # TODO consolidate the two very similar error messages?
            raise ValueError(
                f"Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions less than or equal to the length of the array. "
                f"However indexer={indexer} has {num_single_axis_indexing_expressions} single-axis indexing expressions and array has {arr_ndim} dimensions."
                f"\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
                f"or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
            )
        else:
            # TODO replace ellipses with 0 or more slice(None)s until there are arr_ndim single-axis indexing expressions (so ignoring Nones)
            ...
    else:  # num_ellipses == 0
        if num_single_axis_indexing_expressions != arr_ndim:
            raise ValueError(
                f"Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions equal to the length of the array. "
                f"However indexer={indexer} has {num_single_axis_indexing_expressions} single-axis indexing expressions and array has {arr_ndim} dimensions."
                f"\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
                f"or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
            )
        else:
            return indexer


def apply_indexer(marr: ManifestArray, indexer: T_SimpleIndexer) -> ManifestArray:
    """
    Actually index along each dimension.
    
    Iterates over the axes and applies each indexer one-by-one.
    Encountering a None means inserting a new axis at that position.
    """
    output_arr = marr
    for ind, axis_indexer in enumerate(indexer):
        if axis_indexer is None:
            output_arr = expand_dims(output_arr, axis=ind)
        elif axis_indexer == slice(None):
            # no-op
            pass
        elif isinstance(axis_indexer, int):
            # TODO move other type checking to this stage
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Doesn't support slicing with {indexer}")
    return output_arr
