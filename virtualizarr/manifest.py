from typing import Any, Callable, Dict, Iterable, Mapping, Tuple, TypedDict

import numpy as np

from .types import KerchunkArrRefs, ZArray

HANDLED_ARRAY_FUNCTIONS: Dict[
    str, Callable
] = {}  # populated by the @implements decorators below


class ChunkEntry(TypedDict):
    """
    Information for a single chunk in the manifest.

    Stored in the form `{"path": "s3://bucket/foo.nc", "offset": 100, "length": 100}`.
    """

    path: str
    offset: int
    length: int


class ChunkManifest(Mapping):
    """
    In-memory representation of a single Zarr chunk manifest.

    Stores the manifest in this form:

    {
        "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
        "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
        "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
    }

    See the chunk manifest SPEC proposal in https://github.com/zarr-developers/zarr-specs/issues/287 .
    """

    # TODO make this immutable?

    chunks: Mapping[str, ChunkEntry]
    ndim_chunk_grid: int
    shape_chunk_grid: Tuple[int, ...]

    def __init__(self, chunkentries: Mapping[str, ChunkEntry]):
        self.ndim_chunk_grid, self.shape_chunk_grid = validate_chunk_keys(
            chunkentries.keys()
        )
        validate_chunk_entries(chunkentries.values())

        # TODO allow conversion of strings to `ChunkEntry` objects?
        self._chunks = chunkentries

    @staticmethod
    def from_zarr_json(filepath: str) -> "ChunkManifest":
        """Create a ChunkManifest from a Zarr manifest.json file."""
        raise NotImplementedError()

    def to_zarr_json(self, filepath: str) -> None:
        """Write a ChunkManifest to a Zarr manifest.json file."""
        raise NotImplementedError()


def get_ndim_from_key(key: str) -> int:
    """Get number of dimensions implied by key, e.g. '4.5.6' -> 3"""
    return len(key.split("."))


def validate_chunk_keys(chunk_keys: Iterable[str]) -> Tuple[int, Tuple[int, ...]]:
    # TODO check if all keys have the correct form
    for key in chunk_keys:
        # TODO use a regex instead?
        ...

    # Check if all keys have the same number of dimensions
    first_key, *other_keys = list(chunk_keys)
    ndim = get_ndim_from_key(first_key)
    for key in other_keys:
        other_ndim = get_ndim_from_key(key)
        if other_ndim != ndim:
            raise ValueError(
                f"Inconsistent number of dimensions between chunk key {key} and {first_key}: {other_ndim} vs {ndim}"
            )

    # Check if keys form a complete grid
    # TODO are zarr arrays allowed to have missing chunks?
    shape = ...

    return ndim, shape


def validate_chunk_entries(chunk_entries) -> None:
    ...


class ManifestArray:
    """
    Virtualized array representation of the chunk data in a single Zarr Array.

    Supports concatenation / stacking, but only if the two arrays to be concatenated have the same codecs.

    Cannot be directly altered.

    Implements subset of the array API standard such that it can be wrapped by xarray.
    Doesn't store the zarr array name, zattrs or ARRAY_DIMENSIONS, as instead those can be stored on a wrapping xarray object.
    """

    # TODO how do we forbid variable-length chunks?

    _manifest: ChunkManifest
    _zarray: ZArray

    def __init__(self, zarray: ZArray, chunkmanifest: ChunkManifest) -> None:
        ...

    @staticmethod
    def from_kerchunk_refs(refs: KerchunkArrRefs) -> "ManifestArray":
        ...

    @property
    def chunks(self) -> tuple[int]:
        # TODO do we even need this? The way I implemented concat below I don't think we really do...
        return tuple(self._zarray["chunks"])

    @property
    def dtype(self) -> np.dtype:
        dtype_str = self._zarray["dtype"]
        return np.dtype(dtype_str)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(length) for length in list(self._zarray["shape"]))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape)

    @property
    def T(self) -> "ManifestArray":
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"ManifestArray<shape={self.shape}, dtype={self.dtype}, chunks={self.chunks}>"

    def to_kerchunk_refs(self) -> KerchunkArrRefs:
        # TODO is there enough information to get the attrs and so on here?
        ...

    def to_zarr(self, store) -> None:
        raise NotImplementedError(
            "Requires the chunk manifest ZEP to be formalized before we know what to write out here."
        )

    def __array_function__(self, func, types, args, kwargs) -> Any:
        """
        Hook to teach this class what to do if np.concat etc. is called on it.

        Use this instead of __array_namespace__ so that we don't make promises we can't keep.
        """

        if func not in HANDLED_ARRAY_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle ManifestArray objects
        if not all(issubclass(t, ManifestArray) for t in types):
            return NotImplemented

        return HANDLED_ARRAY_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Any:
        """We have to define this in order to convince xarray that this class is a duckarray, even though we will never support ufuncs."""
        return NotImplemented

    def __array__(self) -> np.ndarray:
        raise NotImplementedError(
            "ManifestArrays can't be converted into numpy arrays or pandas Index objects"
        )


def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""

    def decorator(func):
        HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


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

    # TODO is a codec the same as a compressor?
    first_codec, *other_codecs = [arr._zarray.codec for arr in arrays]
    for codec in other_codecs:
        if codec != first_codec:
            raise NotImplementedError(
                "The ManifestArray class cannot concatenate arrays which were stored using different codecs. "
                "See https://github.com/zarr-developers/zarr-specs/issues/288"
            )

    concatenated_manifest = _concat_manifests(
        [arr._manifest for arr in arrays],
        axis=axis,
    )
    new_shape = ...
    new_zarray = _replace_shape(arrays[0]._zarray, new_shape)

    return ManifestArray(chunkmanifest=concatenated_manifest, zarray=new_zarray)


def _concat_manifests(manifests: Iterable[ChunkManifest], axis: int | Tuple[int, ...]):
    ...


def _replace_shape(zarray: ZArray, new_shape: Tuple[int, ...]) -> ZArray:
    ...


@implements(np.result_type)
def result_type(*arrays_and_dtypes) -> np.dtype:
    """Called by xarray to ensure all arguments to concat have the same dtype."""
    first_dtype, *other_dtypes = [np.dtype(obj) for obj in arrays_and_dtypes]
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError("dtypes not all consistent")
    return first_dtype
