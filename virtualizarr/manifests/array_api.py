from typing import TYPE_CHECKING, Any, Callable, Iterable, cast

import numpy as np

from virtualizarr.zarr import Codec, ceildiv

from .manifest import ChunkManifest

if TYPE_CHECKING:
    from .array import ManifestArray


MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS: dict[
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


def _check_same_codecs(codecs: list[Codec]) -> None:
    first_codec, *other_codecs = codecs
    for codec in other_codecs:
        if codec != first_codec:
            raise NotImplementedError(
                "The ManifestArray class cannot concatenate arrays which were stored using different codecs, "
                f"But found codecs {first_codec} vs {codec} ."
                "See https://github.com/zarr-developers/zarr-specs/issues/288"
            )


def _check_same_chunk_shapes(chunks_list: list[tuple[int, ...]]) -> None:
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
    first_dtype, *other_dtypes = (np.dtype(obj) for obj in arrays_and_dtypes)
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError("dtypes not all consistent")
    return first_dtype


@implements(np.concatenate)
def concatenate(
    arrays: tuple["ManifestArray", ...] | list["ManifestArray"],
    /,
    *,
    axis: int | None = 0,
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

    # do concatenation of entries in manifest
    concatenated_paths = np.concatenate(
        [arr.manifest._paths for arr in arrays],
        axis=axis,
    )
    concatenated_offsets = np.concatenate(
        [arr.manifest._offsets for arr in arrays],
        axis=axis,
    )
    concatenated_lengths = np.concatenate(
        [arr.manifest._lengths for arr in arrays],
        axis=axis,
    )
    concatenated_manifest = ChunkManifest.from_arrays(
        paths=concatenated_paths,
        offsets=concatenated_offsets,
        lengths=concatenated_lengths,
    )

    # chunk shape has not changed, there are just now more chunks along the concatenation axis
    new_zarray = first_arr.zarray.replace(
        shape=tuple(new_shape),
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

    # do stacking of entries in manifest
    stacked_paths = cast(  # `np.stack` apparently is type hinted as if the output could have Any dtype
        np.ndarray[Any, np.dtypes.StringDType],
        np.stack(
            [arr.manifest._paths for arr in arrays],
            axis=axis,
        ),
    )
    stacked_offsets = np.stack(
        [arr.manifest._offsets for arr in arrays],
        axis=axis,
    )
    stacked_lengths = np.stack(
        [arr.manifest._lengths for arr in arrays],
        axis=axis,
    )
    stacked_manifest = ChunkManifest.from_arrays(
        paths=stacked_paths,
        offsets=stacked_offsets,
        lengths=stacked_lengths,
    )

    # chunk shape has changed because a length-1 axis has been inserted
    old_chunks = first_arr.chunks
    new_chunks = list(old_chunks)
    new_chunks.insert(axis, 1)

    new_zarray = first_arr.zarray.replace(
        chunks=tuple(new_chunks),
        shape=tuple(new_shape),
    )

    return ManifestArray(chunkmanifest=stacked_manifest, zarray=new_zarray)


def _check_same_shapes(shapes: list[tuple[int, ...]]) -> None:
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
def broadcast_to(x: "ManifestArray", /, shape: tuple[int, ...]) -> "ManifestArray":
    """
    Broadcasts a ManifestArray to a specified shape, by either adjusting chunk keys or copying chunk manifest entries.
    """

    from .array import ManifestArray

    new_shape = shape

    # check its actually possible to broadcast to this new shape
    mutually_broadcastable_shape = np.broadcast_shapes(x.shape, new_shape)
    if mutually_broadcastable_shape != new_shape:
        # we're not trying to broadcast both shapes to a third shape
        raise ValueError(
            f"array of shape {x.shape} cannot be broadcast to shape {new_shape}"
        )

    # new chunk_shape is old chunk_shape with singleton dimensions pre-pended
    # (chunk shape can never change by more than adding length-1 axes because each chunk represents a fixed number of array elements)
    old_chunk_shape = x.chunks
    new_chunk_shape = _prepend_singleton_dimensions(
        old_chunk_shape, ndim=len(new_shape)
    )

    # find new chunk grid shape by dividing new array shape by new chunk shape
    new_chunk_grid_shape = tuple(
        ceildiv(axis_length, chunk_length)
        for axis_length, chunk_length in zip(new_shape, new_chunk_shape)
    )

    # do broadcasting of entries in manifest
    broadcasted_paths = cast(  # `np.broadcast_to` apparently is type hinted as if the output could have Any dtype
        np.ndarray[Any, np.dtypes.StringDType],
        np.broadcast_to(
            x.manifest._paths,
            shape=new_chunk_grid_shape,
        ),
    )

    broadcasted_offsets = np.broadcast_to(
        x.manifest._offsets,
        shape=new_chunk_grid_shape,
    )
    broadcasted_lengths = np.broadcast_to(
        x.manifest._lengths,
        shape=new_chunk_grid_shape,
    )
    broadcasted_manifest = ChunkManifest.from_arrays(
        paths=broadcasted_paths,
        offsets=broadcasted_offsets,
        lengths=broadcasted_lengths,
    )

    new_zarray = x.zarray.replace(
        chunks=new_chunk_shape,
        shape=new_shape,
    )

    return ManifestArray(chunkmanifest=broadcasted_manifest, zarray=new_zarray)


def _prepend_singleton_dimensions(shape: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Prepend as many new length-1 axes to shape as necessary such that the result has ndim number of axes."""
    n_prepended_dims = ndim - len(shape)
    return tuple([1] * n_prepended_dims + list(shape))


# TODO broadcast_arrays, squeeze, permute_dims


@implements(np.full_like)
def full_like(
    x: "ManifestArray", /, fill_value: bool, *, dtype: np.dtype | None
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
    return _isnan(x.shape)


def _isnan(shape: tuple):
    return np.full(shape=shape, fill_value=False, dtype=np.dtype(bool))
