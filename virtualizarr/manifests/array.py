import warnings
from typing import Any, Tuple, Union

import numpy as np

from ..kerchunk import KerchunkArrRefs
from ..zarr import ZArray
from .array_api import MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS
from .manifest import ChunkManifest


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

        self._zarray = _zarray
        self._manifest = _chunkmanifest

    @classmethod
    def _from_kerchunk_refs(cls, arr_refs: KerchunkArrRefs) -> "ManifestArray":
        from virtualizarr.kerchunk import fully_decode_arr_refs, parse_array_refs

        decoded_arr_refs = fully_decode_arr_refs(arr_refs)

        chunk_dict, zarray, _zattrs = parse_array_refs(decoded_arr_refs)
        manifest = ChunkManifest._from_kerchunk_chunk_dict(chunk_dict)

        obj = object.__new__(cls)
        obj._manifest = manifest
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

    def __repr__(self) -> str:
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

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

    # Everything beyond here is basically just to make this array class wrappable by xarray #

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """We have to define this in order to convince xarray that this class is a duckarray, even though we will never support ufuncs."""
        return NotImplemented

    def __array__(self) -> np.ndarray:
        raise NotImplementedError(
            "ManifestArrays can't be converted into numpy arrays or pandas Index objects"
        )

    def __eq__(  # type: ignore[override]
        self,
        other: Union[int, float, bool, np.ndarray, "ManifestArray"],
    ) -> np.ndarray:
        """
        Element-wise equality checking.

        Returns a numpy array of booleans.
        """
        if isinstance(other, (int, float, bool)):
            # TODO what should this do when comparing against numpy arrays?
            return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))
        elif not isinstance(other, ManifestArray):
            raise TypeError(
                f"Cannot check equality between a ManifestArray and an object of type {type(other)}"
            )

        if self.shape != other.shape:
            raise NotImplementedError("Unsure how to handle broadcasting like this")

        if self.zarray != other.zarray:
            return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))
        else:
            if self.manifest == other.manifest:
                return np.full(shape=self.shape, fill_value=True, dtype=np.dtype(bool))
            else:
                # TODO this doesn't yet do what it should - it simply returns all False if any of the chunk entries are different.
                # What it should do is return True for the locations where the chunk entries are the same.
                warnings.warn(
                    "__eq__ currently is over-cautious, returning an array of all False if any of the chunk entries don't match.",
                    UserWarning,
                )

                # TODO do chunk-wise comparison
                # TODO expand it into an element-wise result
                return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))

    def astype(self, dtype: np.dtype, /, *, copy: bool = True) -> "ManifestArray":
        """Cannot change the dtype, but needed because xarray will call this even when it's a no-op."""
        if dtype != self.dtype:
            raise NotImplementedError()
        else:
            return self

    def __getitem__(
        self,
        key,
        /,
    ) -> "ManifestArray":
        """
        Only supports extremely limited indexing.

        Only here because xarray will apparently attempt to index into its lazy indexing classes even if the operation would be a no-op anyway.
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
