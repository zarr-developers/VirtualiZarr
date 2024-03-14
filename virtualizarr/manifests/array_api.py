from typing import Callable, Dict, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np

from ..zarr import ZArray, Codec
from .manifest import concat_manifests, stack_manifests

if TYPE_CHECKING:
    from .array import ManifestArray


MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS: Dict[
    str, Callable
] = {}  # populated by the @implements decorators below


def implements(numpy_function):
    """Register an __array_function__ implementation for ManifestArray objects."""

    def decorator(func):
        MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def check_combineable_zarr_arrays(arrays: Iterable["ManifestArray"]) -> None:
    """
    The downside of the ManifestArray approach compared to the VirtualZarrArray concatenation proposal is that
    the result must also be a single valid zarr array, implying that the inputs must have the same dtype, codec etc.
    """
    _check_same_dtypes([arr.dtype for arr in arrays])

    # Can't combine different codecs in one manifest
    # see https://github.com/zarr-developers/zarr-specs/issues/288
    _check_same_codecs([arr.zarray.codec for arr in arrays])

    # Would require variable-length chunks ZEP
    _check_same_chunk_shapes([arr.chunks for arr in arrays])


def _check_same_dtypes(dtypes: list[np.dtype]) -> None:
    """Check all the dtypes are the same"""

    first_dtype, *other_dtypes = dtypes
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent dtypes: {other_dtype} vs {first_dtype}"
            )


def _check_same_codecs(codecs: List[Codec]) -> None:
    first_codec, *other_codecs = codecs
    for codec in other_codecs:
        if codec != first_codec:
            raise NotImplementedError(
                "The ManifestArray class cannot concatenate arrays which were stored using different codecs, "
                f"But found codecs {first_codec} vs {codec} ."
                "See https://github.com/zarr-developers/zarr-specs/issues/288"
            )


def _check_same_chunk_shapes(chunks_list: List[Tuple[int, ...]]) -> None:
    """Check all the chunk shapes are the same"""

    first_chunks, *other_chunks_list = chunks_list
    for other_chunks in other_chunks_list:
        if other_chunks != first_chunks:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent chunk shapes: {other_chunks} vs {first_chunks} ."
                "Requires ZEP003 (Variable-length Chunks)."
            )


@implements(np.concatenate)
def concatenate(
    arrays: tuple["ManifestArray", ...] | list["ManifestArray"], /, *, axis: int | None = 0
) -> "ManifestArray":
    """
    Concatenate ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.concat`.
    """
    from .array import ManifestArray

    if axis is None:
        raise NotImplementedError(
            "If axis=None the array API requires flattening, which is a reshape, which can't be implemented on a ManifestArray."
        )
    elif not isinstance(axis, int):
        raise TypeError()

    # ensure dtypes, shapes, codecs etc. are consistent
    check_combineable_zarr_arrays(arrays)

    _check_same_ndims([arr.ndim for arr in arrays])

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    axis = axis % first_arr.ndim

    arr_shapes = [arr.shape for arr in arrays]
    _check_same_shapes_except_on_concat_axis(arr_shapes, axis)

    # find what new array shape must be
    new_length_along_concat_axis = sum([shape[axis] for shape in arr_shapes])
    first_shape, *_ = arr_shapes
    new_shape = list(first_shape)
    new_shape[axis] = new_length_along_concat_axis

    concatenated_manifest = concat_manifests(
        [arr.manifest for arr in arrays],
        axis=axis,
    )

    new_zarray = ZArray(
        chunks=first_arr.chunks,
        compressor=first_arr.zarray.compressor,
        dtype=first_arr.dtype,
        fill_value=first_arr.zarray.fill_value,
        filters=first_arr.zarray.filters,
        shape=new_shape,
        # TODO presumably these things should be checked for consistency across arrays too?
        order=first_arr.zarray.order,
        zarr_format=first_arr.zarray.zarr_format,
    )

    return ManifestArray(chunkmanifest=concatenated_manifest, zarray=new_zarray)


def _check_same_ndims(ndims: list[int]) -> None:
    first_ndim, *other_ndims = ndims
    for other_ndim in other_ndims:
        if other_ndim != first_ndim:
            raise ValueError(
                f"Cannot concatenate arrays with differing number of dimensions: {first_ndim} vs {other_ndim}"
            )


def _check_same_shapes_except_on_concat_axis(shapes: list[tuple[int, ...]], axis: int):
    """Check that shapes are compatible for concatenation"""
    shapes_without_concat_axis = [
        _remove_element_at_position(shape, axis) for shape in shapes
    ]

    first_shape, *other_shapes = shapes_without_concat_axis
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with shapes {[shape for shape in shapes]}"
            )


def _remove_element_at_position(t: tuple[int, ...], pos: int) -> tuple[int, ...]:
    new_l = list(t)
    new_l.pop(pos)
    return tuple(new_l)


@implements(np.result_type)
def result_type(*arrays_and_dtypes) -> np.dtype:
    """Called by xarray to ensure all arguments to concat have the same dtype."""
    first_dtype, *other_dtypes = [np.dtype(obj) for obj in arrays_and_dtypes]
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError("dtypes not all consistent")
    return first_dtype


@implements(np.stack)
def stack(
    arrays: tuple["ManifestArray", ...] | list["ManifestArray"],
    /,
    *,
    axis: int = 0,
) -> "ManifestArray":
    """
    Stack ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.stack`.
    """
    from .array import ManifestArray

    if not isinstance(axis, int):
        raise TypeError()

    # ensure dtypes, shapes, codecs etc. are consistent
    check_combineable_zarr_arrays(arrays)

    _check_same_ndims([arr.ndim for arr in arrays])
    arr_shapes = [arr.shape for arr in arrays]
    _check_same_shapes(arr_shapes)

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    axis = axis % first_arr.ndim

    # find what new array shape must be
    length_along_new_stacked_axis = len(arrays)
    first_shape, *_ = arr_shapes
    new_shape = list(first_shape)
    new_shape.insert(axis, length_along_new_stacked_axis)

    stacked_manifest = stack_manifests(
        [arr.manifest for arr in arrays],
        axis=axis,
    )

    # chunk size has changed because a length-1 axis has been inserted
    old_chunks = first_arr.chunks
    new_chunks = list(old_chunks)
    new_chunks.insert(axis, 1)

    new_zarray = ZArray(
        chunks=new_chunks,
        compressor=first_arr.zarray.compressor,
        dtype=first_arr.dtype,
        fill_value=first_arr.zarray.fill_value,
        filters=first_arr.zarray.filters,
        shape=new_shape,
        # TODO presumably these things should be checked for consistency across arrays too?
        order=first_arr.zarray.order,
        zarr_format=first_arr.zarray.zarr_format,
    )

    return ManifestArray(chunkmanifest=stacked_manifest, zarray=new_zarray)


def _check_same_shapes(shapes: List[Tuple[int, ...]]) -> None:
    first_shape, *other_shapes = shapes
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with differing shapes: {first_shape} vs {other_shape}"
            )
