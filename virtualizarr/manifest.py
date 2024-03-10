import itertools
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    Union,
    cast,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, validator

from .kerchunk import KerchunkArrRefs
from .types import ChunkKey, ZArray

HANDLED_ARRAY_FUNCTIONS: Dict[
    str, Callable
] = {}  # populated by the @implements decorators below


_INTEGER = (
    r"([1-9]+\d*|0)"  # matches 0 or an unsigned integer that does not begin with zero
)
_SEPARATOR = r"\."
_CHUNK_KEY = rf"^{_INTEGER}+({_SEPARATOR}{_INTEGER})*$"  # matches 1 integer, optionally followed by more integers each separated by a separator (i.e. a period)


class ChunkEntry(BaseModel):
    """
    Information for a single chunk in the manifest.

    Stored in the form `{"path": "s3://bucket/foo.nc", "offset": 100, "length": 100}`.
    """

    model_config = ConfigDict(frozen=True)

    path: str  # TODO stricter typing/validation of possible local / remote paths?
    offset: int
    length: int

    @classmethod
    def from_kerchunk(
        cls, path_and_byte_range_info: List[Union[str, int]]
    ) -> "ChunkEntry":
        path, offset, length = path_and_byte_range_info
        return ChunkEntry(path=path, offset=offset, length=length)


class ChunkManifest(BaseModel):
    """
    In-memory representation of a single Zarr chunk manifest.

    Stores the manifest as a dictionary under the .chunks attribute, in this form:

    {
        "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
        "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
        "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
    }

    See the chunk manifest SPEC proposal in https://github.com/zarr-developers/zarr-specs/issues/287 .

    Validation is done when this object is instatiated, and this class is immutable,
    so it's not possible to have a ChunkManifest object that does not represent a complete valid grid of chunks.
    """

    model_config = ConfigDict(frozen=True)

    entries: Mapping[ChunkKey, ChunkEntry]
    # shape_chunk_grid: Tuple[int, ...]  # TODO do we need this for anything?

    @validator("entries")
    def validate_chunks(cls, entries: Any) -> Mapping[ChunkKey, ChunkEntry]:
        validate_chunk_keys(list(entries.keys()))
        
        # TODO what if pydantic adjusts anything during validation?
        return entries

    @property
    def ndim_chunk_grid(self):
        """Number of dimensions in the chunk grid."""
        return get_ndim_from_key(list(self.chunks)[0])

    def __getitem__(self, key: ChunkKey) -> ChunkEntry:
        return self.chunks[key]

    def __iter__(self) -> Iterator[ChunkKey]:
        return iter(self.chunks.keys())

    def __len__(self) -> int:
        return len(self.chunks)

    def dict(self) -> dict[str, dict[str, Union[str, int]]]:
        """Converts the entire manifest to a nested dictionary."""
        return {k: entry.dict() for k, entry in self.entries.items()}

    @staticmethod
    def from_zarr_json(filepath: str) -> "ChunkManifest":
        """Create a ChunkManifest from a Zarr manifest.json file."""
        raise NotImplementedError()

    def to_zarr_json(self, filepath: str) -> None:
        """Write a ChunkManifest to a Zarr manifest.json file."""
        raise NotImplementedError()

    @classmethod
    def from_kerchunk_chunk_dict(cls, kerchunk_chunk_dict) -> "ChunkManifest":
        print(kerchunk_chunk_dict)
        chunkentries = {
            k: ChunkEntry.from_kerchunk(v) for k, v in kerchunk_chunk_dict.items()
        }
        return ChunkManifest(entries=chunkentries)


def get_ndim_from_key(key: str) -> int:
    """Get number of dimensions implied by key, e.g. '4.5.6' -> 3"""
    return len(key.split("."))


def validate_chunk_keys(chunk_keys: Iterable[ChunkKey]) -> Tuple[int, Tuple[int, ...]]:
    # Check if all keys have the correct form
    for key in chunk_keys:
        if not re.match(_CHUNK_KEY, key):
            raise ValueError(f"Invalid format for chunk key: '{key}'")

    # Check if all keys have the same number of dimensions
    first_key, *other_keys = list(chunk_keys)
    ndim = get_ndim_from_key(first_key)
    for key in other_keys:
        other_ndim = get_ndim_from_key(key)
        if other_ndim != ndim:
            raise ValueError(
                f"Inconsistent number of dimensions between chunk key {key} and {first_key}: {other_ndim} vs {ndim}"
            )

    # Check that the keys collectively form a complete grid
    check_keys_form_grid(chunk_keys)


def check_keys_form_grid(chunk_keys: Iterable[ChunkKey]) -> Tuple[int, ...]:
    """Check that the chunk keys collectively form a complete grid"""

    def split(key: ChunkKey) -> Iterable[int]:
        return list(int(i) for i in key.split("."))

    def join(inds: Iterable[int]) -> ChunkKey:
        return cast(ChunkKey, ".".join(str(i) for i in inds))

    # find maximum along each dimension
    zipped_indices = zip(*[split(key) for key in chunk_keys])
    chunk_grid_shape = tuple(
        max(indices_along_one_dim) for indices_along_one_dim in zipped_indices
    )

    # create every possible combination
    all_possible_combos = itertools.product(
        *[range(max + 1) for max in chunk_grid_shape]
    )
    all_required_chunk_keys: set[ChunkKey] = set(
        join(inds) for inds in all_possible_combos
    )

    # check that every possible combination is represented once in the list of chunk keys
    if set(chunk_keys) != all_required_chunk_keys:
        raise ValueError("Chunk keys do not form a complete grid")

    return chunk_grid_shape


def validate_zarray(zarray: ZArray) -> ZArray:
    # TODO actually do this validation
    # can we use pydantic for this?
    return zarray


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
        self._manifest = chunkmanifest
        self._zarray = validate_zarray(zarray)

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
    # ans: codec is compressor + filters
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
