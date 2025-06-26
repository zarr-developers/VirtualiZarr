import warnings
from types import EllipsisType
from typing import Any, Callable, Union

import numpy as np
import xarray as xr
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGrid

import virtualizarr.manifests.utils as utils
from virtualizarr.manifests.array_api import (
    MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS,
    _isnan,
    expand_dims,
)
from virtualizarr.manifests.manifest import ChunkManifest


class ManifestArray:
    """
    Virtualized array representation of the chunk data in a single Zarr Array.

    Supports concatenation / stacking, but only if the two arrays to be concatenated have the same codecs.

    Cannot be directly altered.

    Implements subset of the array API standard such that it can be wrapped by xarray.
    Doesn't store the zarr array name, zattrs or ARRAY_DIMENSIONS, as instead those can be stored on a wrapping xarray object.
    """

    _manifest: ChunkManifest
    _metadata: ArrayV3Metadata

    def __init__(
        self,
        metadata: ArrayV3Metadata | dict,
        chunkmanifest: dict | ChunkManifest,
    ) -> None:
        """
        Create a ManifestArray directly from the metadata of a zarr array and the manifest of chunks.

        Parameters
        ----------
        metadata : dict or ArrayV3Metadata
        chunkmanifest : dict or ChunkManifest
        """

        if isinstance(metadata, ArrayV3Metadata):
            _metadata = metadata
        else:
            # try unpacking the dict
            _metadata = ArrayV3Metadata(**metadata)

        if not isinstance(_metadata.chunk_grid, RegularChunkGrid):
            raise NotImplementedError(
                f"Only RegularChunkGrid is currently supported for chunk size, but got type {type(_metadata.chunk_grid)}"
            )

        if isinstance(chunkmanifest, ChunkManifest):
            _chunkmanifest = chunkmanifest
        elif isinstance(chunkmanifest, dict):
            _chunkmanifest = ChunkManifest(entries=chunkmanifest)
        else:
            raise TypeError(
                f"chunkmanifest arg must be of type ChunkManifest or dict, but got type {type(chunkmanifest)}"
            )

        # TODO check that the metadata shape and chunkmanifest shape are consistent with one another
        # TODO also cover the special case of scalar arrays

        self._metadata = _metadata
        self._manifest = _chunkmanifest

    @property
    def manifest(self) -> ChunkManifest:
        return self._manifest

    @property
    def metadata(self) -> ArrayV3Metadata:
        return self._metadata

    @property
    def chunks(self) -> tuple[int, ...]:
        """
        Individual chunk size by number of elements.
        """
        return self._metadata.chunks

    @property
    def dtype(self) -> np.dtype:
        dtype_str = self.metadata.data_type
        return dtype_str.to_numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Array shape by number of elements along each dimension.
        """
        return self.metadata.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def __repr__(self) -> str:
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

    @property
    def nbytes_virtual(self) -> int:
        """
        Size required to hold these references in memory in bytes.

        Note this is not the size of the referenced array if it were actually loaded into memory,
        this is only the size of the pointers to the chunk locations.
        If you were to load the data into memory it would be ~1e6x larger for 1MB chunks.
        """
        # note: we don't name this method `.nbytes` as we don't want xarray's repr to use it
        return self.manifest.nbytes

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
        if ufunc == np.isnan:
            return _isnan(self.shape)
        return NotImplemented

    def __array__(self, dtype: np.typing.DTypeLike = None) -> np.ndarray:
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
        if isinstance(other, (int, float, bool, np.ndarray)):
            # TODO what should this do when comparing against numpy arrays?
            return np.full(shape=self.shape, fill_value=False, dtype=np.dtype(bool))
        elif not isinstance(other, ManifestArray):
            raise TypeError(
                f"Cannot check equality between a ManifestArray and an object of type {type(other)}"
            )

        if self.shape != other.shape:
            raise NotImplementedError("Unsure how to handle broadcasting like this")

        if not utils.metadata_identical(self.metadata, other.metadata):
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

                # do chunk-wise comparison
                equal_chunk_paths = self.manifest._paths == other.manifest._paths
                equal_chunk_offsets = self.manifest._offsets == other.manifest._offsets
                equal_chunk_lengths = self.manifest._lengths == other.manifest._lengths

                equal_chunks = (
                    equal_chunk_paths & equal_chunk_offsets & equal_chunk_lengths
                )

                if not equal_chunks.all():
                    # TODO expand chunk-wise comparison into an element-wise result instead of just returning all False
                    return np.full(
                        shape=self.shape, fill_value=False, dtype=np.dtype(bool)
                    )
                else:
                    raise RuntimeWarning("Should not be possible to get here")

    def astype(self, dtype: np.dtype, /, *, copy: bool = True) -> "ManifestArray":
        """Cannot change the dtype, but needed because xarray will call this even when it's a no-op."""
        if dtype != self.dtype:
            raise NotImplementedError()
        else:
            return self

    def __getitem__(
        self,
        key: Union[
            int,
            slice,
            EllipsisType,
            None,
            tuple[Union[int, slice, EllipsisType, None, np.ndarray], ...],
            np.ndarray,
        ],
        /,
    ) -> "ManifestArray":
        """
        Only supports extremely limited indexing.

        Only here because xarray will apparently attempt to index into its lazy indexing classes even if the operation would be a no-op anyway.
        """
        from xarray.core.indexing import BasicIndexer

        indexer: tuple[Union[int, slice, EllipsisType, None, np.ndarray], ...]
        # check type is valid
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

        # check value is valid
        indexer = _possibly_expand_trailing_ellipsis(indexer, self.ndim)
        if len(indexer) != self.ndim:
            raise ValueError(
                f"Invalid indexer for array. Indexer length must be less than or equal to the number of dimensions in the array, "
                f"but indexer={indexer} has length {len(indexer)} and array has {ndim} dimensions."
                f"\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
                f"or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
            )

        if all(
            isinstance(axis_indexer, slice) and axis_indexer == slice(None)
            for axis_indexer in indexer
        ):
            # indexer is all slice(None)'s, so this is a no-op
            return self
        else:
            output_arr = self
            for ind, axis_indexer in enumerate(indexer):
                if axis_indexer is None:
                    output_arr = expand_dims(output_arr, axis=ind)
                elif axis_indexer != slice(None):
                    raise NotImplementedError(f"Doesn't support slicing with {indexer}")
            return output_arr

    def rename_paths(
        self,
        new: str | Callable[[str], str],
    ) -> "ManifestArray":
        """
        Rename paths to chunks in this array's manifest.

        Accepts either a string, in which case this new path will be used for all chunks, or
        a function which accepts the old path and returns the new path.

        Parameters
        ----------
        new
            New path to use for all chunks, either as a string, or as a function which accepts and returns strings.

        Returns
        -------
        ManifestArray

        See Also
        --------
        ChunkManifest.rename_paths

        Examples
        --------
        Rename paths to reflect moving the referenced files from local storage to an S3 bucket.

        >>> def local_to_s3_url(old_local_path: str) -> str:
        ...     from pathlib import Path
        ...
        ...     new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"
        ...
        ...     filename = Path(old_local_path).name
        ...     return str(new_s3_bucket_url / filename)
        >>>
        >>> marr.rename_paths(local_to_s3_url)
        """
        renamed_manifest = self.manifest.rename_paths(new)
        return ManifestArray(metadata=self.metadata, chunkmanifest=renamed_manifest)

    def to_virtual_variable(self) -> xr.Variable:
        """
        Create a "virtual" xarray.Variable containing the contents of one zarr array.

        The returned variable will be "virtual", i.e. it will wrap a single ManifestArray object.
        """

        # The xarray data model stores dimension names and arbitrary extra metadata outside of the wrapped array class,
        # so to avoid that information being duplicated we strip it from the ManifestArray before wrapping it.
        dims = self.metadata.dimension_names
        attrs = self.metadata.attributes
        stripped_metadata = utils.copy_and_replace_metadata(
            self.metadata, new_dimension_names=None, new_attributes={}
        )
        stripped_marr = ManifestArray(
            chunkmanifest=self.manifest, metadata=stripped_metadata
        )

        return xr.Variable(
            data=stripped_marr,
            dims=dims,
            attrs=attrs,
        )


def _possibly_expand_trailing_ellipsis(
    indexer: tuple[Union[int, slice, EllipsisType, None, np.ndarray], ...],
    ndim: int,
):
    """
    Allows for passing indexers <= the shape of the array, so long as they end with an ellipsis.

    For example:

    marr[3, slice(2), ...]

    where marr.ndim => 3.
    """
    final_dim_indexer = indexer[-1]
    if final_dim_indexer == ...:
        if len(indexer) > ndim:
            raise ValueError(
                f"Invalid indexer for array. Indexer length must be less than or equal to the number of dimensions in the array, "
                f"but indexer={indexer} has length {len(indexer)} and array has {ndim} dimensions."
                f"\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
                f"or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
            )

        extra_slices_needed = ndim - (len(indexer) - 1)
        *indexer_as_list, ellipsis = indexer
        return tuple(tuple(indexer_as_list) + (slice(None),) * extra_slices_needed)
    else:
        return indexer
