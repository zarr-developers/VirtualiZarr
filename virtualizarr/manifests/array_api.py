import itertools
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Tuple, Union, cast

import numpy as np

from ..zarr import Codec, ZArray
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


def _check_combineable_zarr_arrays(arrays: Iterable["ManifestArray"]) -> None:
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


@implements(np.result_type)
def result_type(*arrays_and_dtypes) -> np.dtype:
    """Called by xarray to ensure all arguments to concat have the same dtype."""
    first_dtype, *other_dtypes = [np.dtype(obj) for obj in arrays_and_dtypes]
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError("dtypes not all consistent")
    return first_dtype


@implements(np.concatenate)
def concatenate(
    arrays: Union[tuple["ManifestArray", ...], list["ManifestArray"]],
    /,
    *,
    axis: Union[int, None] = 0,
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
    _check_combineable_zarr_arrays(arrays)

    _check_same_ndims([arr.ndim for arr in arrays])

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    if axis < 0:
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


@implements(np.stack)
def stack(
    arrays: Union[tuple["ManifestArray", ...], list["ManifestArray"]],
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
    _check_combineable_zarr_arrays(arrays)

    _check_same_ndims([arr.ndim for arr in arrays])
    arr_shapes = [arr.shape for arr in arrays]
    _check_same_shapes(arr_shapes)

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    if axis < 0:
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


@implements(np.expand_dims)
def expand_dims(x: "ManifestArray", /, *, axis: int = 0) -> "ManifestArray":
    """Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by axis."""
    # this is just a special case of stacking
    return stack([x], axis=axis)


@implements(np.broadcast_to)
def broadcast_to(x: "ManifestArray", /, shape: Tuple[int, ...]) -> "ManifestArray":
    """
    Broadcasts an array to a specified shape, by either manipulating chunk keys or copying chunk manifest entries.
    """

    if len(x.shape) > len(shape):
        raise ValueError("input operand has more dimensions than allowed")

    # numpy broadcasting algorithm requires us to start by comparing the length of the final axes and work backwards
    result = x
    for axis, d, d_requested in itertools.zip_longest(
        reversed(range(len(shape))), reversed(x.shape), reversed(shape), fillvalue=None
    ):
        # len(shape) check above ensures this can't be type None
        d_requested = cast(int, d_requested)

        if d == d_requested:
            pass
        elif d is None:
            if result.shape == ():
                # scalars are a special case because their manifests already have a chunk key with one dimension
                # see https://github.com/TomNicholas/VirtualiZarr/issues/100#issuecomment-2097058282
                result = _broadcast_scalar(result, new_axis_length=d_requested)
            else:
                # stack same array upon itself d_requested number of times, which inserts a new axis at axis=0
                result = stack([result] * d_requested, axis=0)
        elif d == 1:
            # concatenate same array upon itself d_requested number of times along existing axis
            result = concatenate([result] * d_requested, axis=axis)
        else:
            raise ValueError(
                f"Array with shape {x.shape} cannot be broadcast to shape {shape}"
            )

    return result


def _broadcast_scalar(x: "ManifestArray", new_axis_length: int) -> "ManifestArray":
    """
    Add an axis to a scalar ManifestArray, but without adding a new axis to the keys of the chunk manifest.

    This is not the same as concatenation, because there is no existing axis along which to concatenate.
    It's also not the same as stacking, because we don't want to insert a new axis into the chunk keys.

    Scalars are a special case because their manifests still have a chunk key with one dimension.
    See https://github.com/TomNicholas/VirtualiZarr/issues/100#issuecomment-2097058282
    """

    from .array import ManifestArray

    new_shape = (new_axis_length,)
    new_chunks = (new_axis_length,)

    concatenated_manifest = concat_manifests(
        [x.manifest] * new_axis_length,
        axis=0,
    )

    new_zarray = ZArray(
        chunks=new_chunks,
        compressor=x.zarray.compressor,
        dtype=x.dtype,
        fill_value=x.zarray.fill_value,
        filters=x.zarray.filters,
        shape=new_shape,
        order=x.zarray.order,
        zarr_format=x.zarray.zarr_format,
    )

    return ManifestArray(chunkmanifest=concatenated_manifest, zarray=new_zarray)


# TODO broadcast_arrays, squeeze, permute_dims


@implements(np.full_like)
def full_like(
    x: "ManifestArray", /, fill_value: bool, *, dtype: Union[np.dtype, None]
) -> np.ndarray:
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.

    Returns a numpy array instead of a ManifestArray.

    Only implemented to get past some checks deep inside xarray, see https://github.com/TomNicholas/VirtualiZarr/issues/29.
    """
    return np.full(
        shape=x.shape,
        fill_value=fill_value,
        dtype=dtype if dtype is not None else x.dtype,
    )


@implements(np.isnan)
def isnan(x: "ManifestArray", /) -> np.ndarray:
    """
    Returns a numpy array of all False.

    Only implemented to get past some checks deep inside xarray, see https://github.com/TomNicholas/VirtualiZarr/issues/29.
    """
    return np.full(
        shape=x.shape,
        fill_value=False,
        dtype=np.dtype(bool),
    )
