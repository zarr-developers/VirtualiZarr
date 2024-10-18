import dataclasses
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Dict, NewType, Tuple, TypedDict, cast

import numpy as np

from virtualizarr.types import ChunkKey

_INTEGER = (
    r"([1-9]+\d*|0)"  # matches 0 or an unsigned integer that does not begin with zero
)
_SEPARATOR = r"\."
_CHUNK_KEY = rf"^{_INTEGER}+({_SEPARATOR}{_INTEGER})*$"  # matches 1 integer, optionally followed by more integers each separated by a separator (i.e. a period)


class ChunkDictEntry(TypedDict):
    path: str
    offset: int
    length: int


ChunkDict = NewType("ChunkDict", dict[ChunkKey, ChunkDictEntry])


@dataclasses.dataclass(frozen=True)
class ChunkEntry:
    """
    Information for a single chunk in the manifest.

    Stored in the form `{"path": "s3://bucket/foo.nc", "offset": 100, "length": 100}`.
    """

    path: str  # TODO stricter typing/validation of possible local / remote paths?
    offset: int
    length: int

    @classmethod
    def from_kerchunk(
        cls, path_and_byte_range_info: tuple[str] | tuple[str, int, int]
    ) -> "ChunkEntry":
        from upath import UPath

        if len(path_and_byte_range_info) == 1:
            path = path_and_byte_range_info[0]
            offset = 0
            length = UPath(path).stat().st_size
        else:
            path, offset, length = path_and_byte_range_info
        return ChunkEntry(path=path, offset=offset, length=length)

    def to_kerchunk(self) -> tuple[str, int, int]:
        """Write out in the format that kerchunk uses for chunk entries."""
        return (self.path, self.offset, self.length)

    def dict(self) -> ChunkDictEntry:
        return ChunkDictEntry(
            path=self.path,
            offset=self.offset,
            length=self.length,
        )


class ChunkManifest:
    """
    In-memory representation of a single Zarr chunk manifest.

    Stores the manifest internally as numpy arrays, so the most efficient way to create this object is via the `.from_arrays` constructor classmethod.

    The manifest can be converted to or from a dictionary which looks like this

        {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }

    using the .__init__() and .dict() methods, so users of this class can think of the manifest as if it were a dict mapping zarr chunk keys to byte ranges.

    (See the chunk manifest SPEC proposal in https://github.com/zarr-developers/zarr-specs/issues/287.)

    Validation is done when this object is instantiated, and this class is immutable,
    so it's not possible to have a ChunkManifest object that does not represent a valid grid of chunks.
    """

    _paths: np.ndarray[Any, np.dtypes.StringDType]
    _offsets: np.ndarray[Any, np.dtype[np.uint64]]
    _lengths: np.ndarray[Any, np.dtype[np.uint64]]

    def __init__(self, entries: dict, shape: tuple[int, ...] | None = None) -> None:
        """
        Create a ChunkManifest from a dictionary mapping zarr chunk keys to byte ranges.

        Parameters
        ----------
        entries: dict
            Chunk keys and byte range information, as a dictionary of the form

                {
                    "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
                    "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
                    "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
                    "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
                }
        """
        if shape is None and not entries:
            raise ValueError("need a chunk grid shape if no chunks given")

        # TODO do some input validation here first?
        validate_chunk_keys(entries.keys())

        if shape is None:
            shape = get_chunk_grid_shape(entries.keys())

        # Initializing to empty implies that entries with path='' are treated as missing chunks
        paths = cast(  # `np.empty` apparently is type hinted as if the output could have Any dtype
            np.ndarray[Any, np.dtypes.StringDType],
            np.empty(shape=shape, dtype=np.dtypes.StringDType()),
        )
        offsets = np.empty(shape=shape, dtype=np.dtype("uint64"))
        lengths = np.empty(shape=shape, dtype=np.dtype("uint64"))

        # populate the arrays
        for key, entry in entries.items():
            try:
                path, offset, length = entry.values()
                entry = ChunkEntry(path=path, offset=offset, length=length)
            except (ValueError, TypeError) as e:
                msg = (
                    "Each chunk entry must be of the form dict(path=<str>, offset=<int>, length=<int>), "
                    f"but got {entry}"
                )
                raise ValueError(msg) from e

            split_key = split(key)
            paths[split_key] = entry.path
            offsets[split_key] = entry.offset
            lengths[split_key] = entry.length

        self._paths = paths
        self._offsets = offsets
        self._lengths = lengths

    @classmethod
    def from_arrays(
        cls,
        paths: np.ndarray[Any, np.dtypes.StringDType],
        offsets: np.ndarray[Any, np.dtype[np.uint64]],
        lengths: np.ndarray[Any, np.dtype[np.uint64]],
    ) -> "ChunkManifest":
        """
        Create manifest directly from numpy arrays containing the path and byte range information.

        Useful if you want to avoid the memory overhead of creating an intermediate dictionary first,
        as these 3 arrays are what will be used internally to store the references.

        Parameters
        ----------
        paths: np.ndarray
        offsets: np.ndarray
        lengths: np.ndarray
        """

        # check types
        if not isinstance(paths, np.ndarray):
            raise TypeError(f"paths must be a numpy array, but got type {type(paths)}")
        if not isinstance(offsets, np.ndarray):
            raise TypeError(
                f"offsets must be a numpy array, but got type {type(offsets)}"
            )
        if not isinstance(lengths, np.ndarray):
            raise TypeError(
                f"lengths must be a numpy array, but got type {type(lengths)}"
            )

        # check dtypes
        if paths.dtype != np.dtypes.StringDType():  # type: ignore[attr-defined]
            raise ValueError(
                f"paths array must have a numpy variable-length string dtype, but got dtype {paths.dtype}"
            )
        if offsets.dtype != np.dtype("uint64"):
            raise ValueError(
                f"offsets array must have 64-bit unsigned integer dtype, but got dtype {offsets.dtype}"
            )
        if lengths.dtype != np.dtype("uint64"):
            raise ValueError(
                f"lengths array must have 64-bit unsigned integer dtype, but got dtype {lengths.dtype}"
            )

        # check shapes
        shape = paths.shape
        if offsets.shape != shape:
            raise ValueError(
                f"Shapes of the arrays must be consistent, but shapes of paths array and offsets array do not match: {paths.shape} vs {offsets.shape}"
            )
        if lengths.shape != shape:
            raise ValueError(
                f"Shapes of the arrays must be consistent, but shapes of paths array and lengths array do not match: {paths.shape} vs {lengths.shape}"
            )

        obj = object.__new__(cls)
        obj._paths = paths
        obj._offsets = offsets
        obj._lengths = lengths

        return obj

    @property
    def ndim_chunk_grid(self) -> int:
        """
        Number of dimensions in the chunk grid.

        Not the same as the dimension of an array backed by this chunk manifest.
        """
        return self._paths.ndim

    @property
    def shape_chunk_grid(self) -> tuple[int, ...]:
        """
        Number of separate chunks along each dimension.

        Not the same as the shape of an array backed by this chunk manifest.
        """
        return self._paths.shape

    def __repr__(self) -> str:
        return f"ChunkManifest<shape={self.shape_chunk_grid}>"

    def __getitem__(self, key: ChunkKey) -> ChunkEntry:
        indices = split(key)
        path = self._paths[indices]
        offset = self._offsets[indices]
        length = self._lengths[indices]
        return ChunkEntry(path=path, offset=offset, length=length)

    def __iter__(self) -> Iterator[ChunkKey]:
        # TODO make this work for numpy arrays
        raise NotImplementedError()
        # return iter(self._paths.keys())

    def __len__(self) -> int:
        return self._paths.size

    def dict(self) -> ChunkDict:  # type: ignore[override]
        """
        Convert the entire manifest to a nested dictionary.

        The returned dict will be of the form

        {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }

        Entries whose path is an empty string will be interpreted as missing chunks and omitted from the dictionary.
        """

        coord_vectors = np.mgrid[
            tuple(slice(None, length) for length in self.shape_chunk_grid)
        ]

        d = {
            join(inds): dict(
                path=path.item(), offset=offset.item(), length=length.item()
            )
            for *inds, path, offset, length in np.nditer(
                [*coord_vectors, self._paths, self._offsets, self._lengths],
                flags=("refs_ok",),
            )
            if path.item() != ""  # don't include entry if path='' (i.e. empty chunk)
        }

        return cast(
            ChunkDict,
            d,
        )

    def __eq__(self, other: Any) -> bool:
        """Two manifests are equal if all of their entries are identical."""
        paths_equal = (self._paths == other._paths).all()
        offsets_equal = (self._offsets == other._offsets).all()
        lengths_equal = (self._lengths == other._lengths).all()
        return paths_equal and offsets_equal and lengths_equal

    @classmethod
    def from_zarr_json(cls, filepath: str) -> "ChunkManifest":
        """Create a ChunkManifest from a Zarr manifest.json file."""

        with open(filepath, "r") as manifest_file:
            entries = json.load(manifest_file)

        return cls(entries=entries)

    def to_zarr_json(self, filepath: str) -> None:
        """Write the manifest to a Zarr manifest.json file."""
        entries = self.dict()
        with open(filepath, "w") as json_file:
            json.dump(entries, json_file, indent=4, separators=(", ", ": "))

    @classmethod
    def _from_kerchunk_chunk_dict(
        cls,
        # The type hint requires `Dict` instead of `dict` due to
        # the conflicting ChunkManifest.dict method.
        kerchunk_chunk_dict: Dict[ChunkKey, str | tuple[str] | tuple[str, int, int]],
    ) -> "ChunkManifest":
        chunk_entries: dict[ChunkKey, ChunkDictEntry] = {}
        for k, v in kerchunk_chunk_dict.items():
            if isinstance(v, (str, bytes)):
                raise NotImplementedError(
                    "Reading inlined reference data is currently not supported. [ToDo]"
                )
            elif not isinstance(v, (tuple, list)):
                raise TypeError(f"Unexpected type {type(v)} for chunk value: {v}")
            chunk_entries[k] = ChunkEntry.from_kerchunk(v).dict()
        return ChunkManifest(entries=chunk_entries)

    def rename_paths(
        self,
        new: str | Callable[[str], str],
    ) -> "ChunkManifest":
        """
        Rename paths to chunks in this manifest.

        Accepts either a string, in which case this new path will be used for all chunks, or
        a function which accepts the old path and returns the new path.

        Parameters
        ----------
        new
            New path to use for all chunks, either as a string, or as a function which accepts and returns strings.

        Returns
        -------
        manifest

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

        >>> manifest.rename_paths(local_to_s3_url)

        See Also
        --------
        ManifestArray.rename_paths
        """
        if isinstance(new, str):
            renamed_paths = np.full_like(self._paths, fill_value=new)
        elif callable(new):
            vectorized_rename_fn = np.vectorize(new, otypes=[np.dtypes.StringDType()])  # type: ignore[attr-defined]
            renamed_paths = vectorized_rename_fn(self._paths)
        else:
            raise TypeError(
                f"Argument 'new' must be either a string or a callable that accepts and returns strings, but got type {type(new)}"
            )

        return self.from_arrays(
            paths=renamed_paths,
            offsets=self._offsets,
            lengths=self._lengths,
        )


def split(key: ChunkKey) -> Tuple[int, ...]:
    return tuple(int(i) for i in key.split("."))


def join(inds: Iterable[Any]) -> ChunkKey:
    return cast(ChunkKey, ".".join(str(i) for i in list(inds)))


def get_ndim_from_key(key: str) -> int:
    """Get number of dimensions implied by key, e.g. '4.5.6' -> 3"""
    return len(key.split("."))


def validate_chunk_keys(chunk_keys: Iterable[ChunkKey]):
    if not chunk_keys:
        return

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


def get_chunk_grid_shape(chunk_keys: Iterable[ChunkKey]) -> tuple[int, ...]:
    # find max chunk index along each dimension
    zipped_indices = zip(*[split(key) for key in chunk_keys])
    chunk_grid_shape = tuple(
        max(indices_along_one_dim) + 1 for indices_along_one_dim in zipped_indices
    )
    return chunk_grid_shape
