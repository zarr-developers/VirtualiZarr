import re
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np

from ..kerchunk import KerchunkArrRefs
from ..zarr import Codec, ZArray
from .manifest import _CHUNK_KEY, ChunkManifest, concat_manifests, stack_manifests

MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS: Dict[
    str, Callable
] = {}  # populated by the @implements decorators below


class ManifestArray:
    """
    Virtualized array representation of the chunk data in a single Zarr Array.

    Supports concatenation / stacking, but only if the two arrays to be concatenated have the same codecs.

    Cannot be directly altered.

    Implements subset of the array API standard such that it can be wrapped by xarray.
    Doesn't store the zarr array name, zattrs or ARRAY_DIMENSIONS, as instead those can be stored on a wrapping xarray object.
    """

    _manifest: ChunkManifest
    _zarray: ZArray

    def __init__(
        self,
        zarray: Union[ZArray, dict],
        chunkmanifest: Union[dict, ChunkManifest],
    ) -> None:
        """
        Create a ManifestArray directly from the .zarray information of a zarr array and the manifest of chunks.

        Parameters
        ----------
        zarray : dict or ZArray
        chunkmanifest : dict or ChunkManifest
        """

        if isinstance(zarray, ZArray):
            _zarray = zarray
        else:
            # try unpacking the dict
            _zarray = ZArray(**zarray)

        if isinstance(chunkmanifest, ChunkManifest):
            _chunkmanifest = chunkmanifest
        elif isinstance(chunkmanifest, dict):
            _chunkmanifest = ChunkManifest(entries=chunkmanifest)
        else:
            raise TypeError(
                f"chunkmanifest arg must be of type ChunkManifest, but got type {type(chunkmanifest)}"
            )

        # Check that the chunk grid implied by zarray info is consistent with shape implied by chunk keys in manifest
        if _zarray.shape_chunk_grid != _chunkmanifest.shape_chunk_grid:
            raise ValueError(
                f"Inconsistent chunk grid shape between zarray info and manifest: {_zarray.shape_chunk_grid} vs {_chunkmanifest.shape_chunk_grid}"
            )

        self._zarray = _zarray
        self._manifest = _chunkmanifest

    @classmethod
    def from_kerchunk_refs(cls, arr_refs: KerchunkArrRefs) -> "ManifestArray":
        from virtualizarr.kerchunk import fully_decode_arr_refs

        decoded_arr_refs = fully_decode_arr_refs(arr_refs)

        zarray = ZArray.from_kerchunk_refs(decoded_arr_refs[".zarray"])

        kerchunk_chunk_dict = {
            k: v for k, v in decoded_arr_refs.items() if re.match(_CHUNK_KEY, k)
        }
        chunkmanifest = ChunkManifest.from_kerchunk_chunk_dict(kerchunk_chunk_dict)

        obj = object.__new__(cls)
        obj._manifest = chunkmanifest
        obj._zarray = zarray

        return obj

    @property
    def manifest(self) -> ChunkManifest:
        return self._manifest

    @property
    def zarray(self) -> ZArray:
        return self._zarray

    @property
    def chunks(self) -> Tuple[int, ...]:
        return tuple(self.zarray.chunks)

    @property
    def dtype(self) -> np.dtype:
        dtype_str = self.zarray.dtype
        return np.dtype(dtype_str)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(length) for length in list(self.zarray.shape))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def T(self) -> "ManifestArray":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

    def to_kerchunk_refs(self) -> KerchunkArrRefs:
        # TODO is there enough information to get the attrs and so on here?
        raise NotImplementedError()

    def to_zarr(self, store) -> None:
        raise NotImplementedError(
            "Requires the chunk manifest ZEP to be formalized before we know what to write out here."
        )

    def __array_function__(self, func, types, args, kwargs) -> Any:
        """
        Hook to teach this class what to do if np.concat etc. is called on it.

        Use this instead of __array_namespace__ so that we don't make promises we can't keep.
        """

        if func not in MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle ManifestArray objects
        if not all(issubclass(t, ManifestArray) for t in types):
            return NotImplemented

        return MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """We have to define this in order to convince xarray that this class is a duckarray, even though we will never support ufuncs."""
        return NotImplemented

    def __array__(self) -> np.ndarray:
        raise NotImplementedError(
            "ManifestArrays can't be converted into numpy arrays or pandas Index objects"
        )

    def __getitem__(
        self,
        key,
        /,
    ) -> "ManifestArray":
        """
        Only supports extremely limited indexing.

        I only added this method because xarray will apparently attempt to index into its lazy indexing classes even if the operation would be a no-op anyway.
        """
        from xarray.core.indexing import BasicIndexer

        if isinstance(key, BasicIndexer):
            indexer = key.tuple
        else:
            indexer = key

        indexer = _possibly_expand_trailing_ellipsis(key, self.ndim)

        if len(indexer) != self.ndim:
            raise ValueError(
                f"Invalid indexer for array with ndim={self.ndim}: {indexer}"
            )

        if all(
            isinstance(axis_indexer, slice) and axis_indexer == slice(None)
            for axis_indexer in indexer
        ):
            # indexer is all slice(None)'s, so this is a no-op
            print("__getitem__ called with a no-op")
            return self
        else:
            raise NotImplementedError(f"Doesn't support slicing with {indexer}")


def _possibly_expand_trailing_ellipsis(key, ndim: int):
    if key[-1] == ...:
        extra_slices_needed = ndim - (len(key) - 1)
        *indexer, ellipsis = key
        return tuple(tuple(indexer) + (slice(None),) * extra_slices_needed)
    else:
        return key


def implements(numpy_function):
    """Register an __array_function__ implementation for ManifestArray objects."""

    def decorator(func):
        MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def check_combineable_zarr_arrays(arrays: Iterable[ManifestArray]) -> None:
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
    arrays: tuple[ManifestArray, ...] | list[ManifestArray], /, *, axis: int | None = 0
) -> ManifestArray:
    """
    Concatenate ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.concat`.
    """
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
    arrays: tuple[ManifestArray, ...] | list[ManifestArray],
    /,
    *,
    axis: int = 0,
) -> ManifestArray:
    """
    Stack ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.stack`.
    """
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
