import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np
import xarray as xr
from zarr.core.metadata.v3 import ArrayV3Metadata

import virtualizarr.manifests.utils as utils
from virtualizarr.manifests.array_api import (
    MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS,
    _isnan,
)
from virtualizarr.manifests.indexing import T_Indexer, index
from virtualizarr.manifests.manifest import ChunkManifest
from virtualizarr.manifests.utils import ChunkKeySeparator
from virtualizarr.utils import determine_chunk_grid_shape

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import RegularChunkGridMetadata
else:
    try:
        from zarr.core.metadata.v3 import RegularChunkGridMetadata  # zarr-python>3.1.6
    except ImportError:
        from zarr.core.metadata.v3 import (
            RegularChunkGrid as RegularChunkGridMetadata,  # zarr-python<=3.1.6
        )


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

        if not isinstance(_metadata.chunk_grid, RegularChunkGridMetadata):
            raise NotImplementedError(
                f"Only RegularChunkGrid is currently supported for chunk size, but got type {type(_metadata.chunk_grid)}"
            )

        if isinstance(chunkmanifest, ChunkManifest):
            _chunkmanifest = chunkmanifest
        elif isinstance(chunkmanifest, dict):
            separator = cast(
                ChunkKeySeparator,
                getattr(_metadata.chunk_key_encoding, "separator", "."),
            )
            _chunkmanifest = ChunkManifest(entries=chunkmanifest, separator=separator)
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
    def dtype(self) -> np.dtype:
        """The native dtype of the data (typically a numpy dtype)"""
        zdtype = self.metadata.data_type
        dtype = zdtype.to_native_dtype()
        return dtype

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
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.metadata.chunks}>"

    @property
    def nbytes_virtual(self) -> int:
        """
        The total number of bytes required to hold these virtual references in memory in bytes.

        Notes
        -----
        This is not the size of the referenced array if it were actually loaded into memory (use `.nbytes`),
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """We have to define this in order to convince xarray that this class is a duckarray, even though we will never support ufuncs."""
        if ufunc == np.isnan:
            return _isnan(self.shape)
        return NotImplemented

    def __array__(
        self, dtype: np.typing.DTypeLike | None = None, copy: bool | None = None
    ) -> np.ndarray:
        raise NotImplementedError(
            "ManifestArray holds virtual references to chunks in archival files and "
            "cannot be converted into a numpy array or pandas Index. This usually means "
            "an xarray operation (e.g. alignment, a value comparison during concat/merge, "
            "or building a repr) tried to read the array's values. To make the values "
            "available, either pass the variable's name in `loadable_variables` when "
            "opening so it is read into memory, or write the virtual dataset to a "
            "Zarr/Icechunk store and reopen it."
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

        if not self.metadata.to_dict() == other.metadata.to_dict():
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
                equal_chunks = self.manifest.elementwise_eq(other.manifest)

                if not equal_chunks.all():
                    # TODO expand chunk-wise comparison into an element-wise result instead of just returning all False
                    return np.full(
                        shape=self.shape, fill_value=False, dtype=np.dtype(bool)
                    )
                else:
                    raise RuntimeWarning("Should not be possible to get here")

    def astype(self, dtype: np.dtype, /, *, copy: bool = True) -> "ManifestArray":
        """Cannot change the dtype, but needed because xarray will call this even when it's a no-op."""
        if not np.issubdtype(self.dtype, dtype):
            raise NotImplementedError()
        else:
            return self

    def __getitem__(
        self,
        key: T_Indexer,
        /,
    ) -> "ManifestArray":
        """
        Index into this ManifestArray, returning a new ManifestArray view over a subset of chunks.

        Supports only chunk-aligned selections. A ManifestArray only stores references to where
        each chunk's bytes live, never their decoded values, so any indexer that would split into
        the interior of a chunk would require loading the underlying data — which defeats the
        point of a virtual array. Selections that would do so raise ``SubChunkIndexingError``
        (a ``ValueError`` subclass); this is a permanent constraint, not a missing feature.

        Supported indexers (and tuples thereof):

        - ``Ellipsis`` and ``None`` — no-ops and new-axis insertion.
        - ``slice`` with ``step == 1`` whose start and stop land on chunk boundaries
          (``stop == axis_length`` is also allowed, so a partial final chunk can be selected).
          Slice indexers preserve the axis.
        - ``int`` — drops the indexed axis, following numpy / array-API semantics. Only legal
          when ``chunk_size == 1`` along that axis; otherwise picking a single element would
          require splitting a chunk.
        - Slice along the largest-stride storage axis of an **uncompressed** array that fits
          entirely within one source chunk — handled by rewriting the chunk reference's byte
          offset/length rather than splitting bytes. Useful for picking a single timestep from
          a multi-row chunk on a parser like the netCDF3 one. The eligible-axis is axis 0 for
          a plain ``[BytesCodec]`` array (C-order) or axis ``order[0]`` of a prepended
          ``[TransposeCodec(order=...), BytesCodec]`` (e.g. the last axis for F-order).

        Anything else — fancy indexing with arrays, misaligned slices, ``step != 1`` —
        raises ``SubChunkIndexingError`` or ``NotImplementedError``.

        Parameters
        ----------
        key
            A basic indexer or tuple of basic indexers, one per array axis (with ``Ellipsis``
            and ``None`` allowed as per the array API).

        Returns
        -------
        ManifestArray
            A new array whose ``ChunkManifest`` references only the selected chunks.
        """
        return index(self, key)

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
        >>> marr.rename_paths(local_to_s3_url)  # doctest: +SKIP
        """
        renamed_manifest = self.manifest.rename_paths(new)
        return ManifestArray(metadata=self.metadata, chunkmanifest=renamed_manifest)

    def with_fill_value_only(self, fill_value: Any) -> "ManifestArray":
        """
        Return a new ManifestArray with the same schema (shape, chunks, codecs,
        dimension names, attributes) as this one, but with an empty chunk
        manifest and the given ``fill_value``.

        Reads from any chunk in the result return ``fill_value`` (see the Zarr V3
        spec for missing-chunk semantics). This is useful as a typed placeholder
        for a variable that is absent from one source but present in others — e.g.
        concatenating with real data along a new axis without materializing chunks.

        Parameters
        ----------
        fill_value
            The scalar value to store on the metadata; every read from the
            resulting array returns this value.
        """
        # dataclasses.replace bypasses the to_dict/from_dict roundtrip used in
        # copy_and_replace_metadata, which can't accept raw NaN scalars (to_dict
        # serializes NaN to the JSON string "NaN")
        new_metadata = dataclasses.replace(self.metadata, fill_value=fill_value)
        empty_manifest = ChunkManifest(
            entries={},
            shape=determine_chunk_grid_shape(self.shape, self.metadata.chunks),
        )
        return ManifestArray(metadata=new_metadata, chunkmanifest=empty_manifest)

    def _reindex_axis(
        self, axis: int, chunk_map: list[int | None], new_size: int
    ) -> "ManifestArray":
        """
        Return a new ManifestArray with the chunk grid along ``axis`` remapped.

        Each entry of ``chunk_map`` is the source chunk index to copy into that
        target chunk slot, or ``None`` for a missing (null-path) chunk that reads
        back as ``fill_value``. ``new_size`` is the new length of ``axis`` in
        elements; the chunk size is unchanged.
        """
        from virtualizarr.manifests.manifest import (
            MISSING_CHUNK_PATH,
            ChunkManifest,
        )
        from virtualizarr.manifests.utils import copy_and_replace_metadata

        manifest = self.manifest
        src_paths = manifest._paths
        src_offsets = manifest._offsets
        src_lengths = manifest._lengths

        new_grid_shape = list(src_paths.shape)
        new_grid_shape[axis] = len(chunk_map)

        new_paths = np.full(
            new_grid_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType()
        )
        new_offsets = np.zeros(new_grid_shape, dtype=np.uint64)
        new_lengths = np.zeros(new_grid_shape, dtype=np.uint64)

        new_inlined: dict[tuple[int, ...], bytes] = {}
        for new_idx, src_chunk in enumerate(chunk_map):
            if src_chunk is None:
                continue  # leave this slab as missing/null
            src_slice: list[Any] = [slice(None)] * src_paths.ndim
            src_slice[axis] = src_chunk
            dst_slice: list[Any] = [slice(None)] * src_paths.ndim
            dst_slice[axis] = new_idx
            new_paths[tuple(dst_slice)] = src_paths[tuple(src_slice)]
            new_offsets[tuple(dst_slice)] = src_offsets[tuple(src_slice)]
            new_lengths[tuple(dst_slice)] = src_lengths[tuple(src_slice)]
            # re-key any inlined chunks that lived in this source slab
            for key, data in manifest._inlined.items():
                if key[axis] == src_chunk:
                    shifted = list(key)
                    shifted[axis] = new_idx
                    new_inlined[tuple(shifted)] = data

        new_manifest = ChunkManifest.from_arrays(
            paths=new_paths,
            offsets=new_offsets,
            lengths=new_lengths,
            validate_paths=False,
            inlined=new_inlined if new_inlined else None,
        )

        new_shape = list(self.shape)
        new_shape[axis] = new_size
        new_metadata = copy_and_replace_metadata(
            old_metadata=self.metadata, new_shape=new_shape
        )

        return ManifestArray(chunkmanifest=new_manifest, metadata=new_metadata)

    def to_virtual_variable(self) -> xr.Variable:
        """
        Create a "virtual" xarray.Variable containing the contents of one zarr array.

        The returned variable will be "virtual", i.e. it will wrap a single ManifestArray object.
        """

        # The xarray data model stores dimension names and arbitrary extra metadata outside of the wrapped array class,
        # so to avoid that information being duplicated we strip it from the ManifestArray before wrapping it.
        if self.metadata.dimension_names is not None:
            dims = self.metadata.dimension_names
        elif self.ndim == 0:
            dims = ()
        else:
            raise ValueError(
                f"Cannot create virtual variable from {self.ndim}-dimensional array without dimension names."
            )
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
