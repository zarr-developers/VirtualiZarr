import json
import re
from typing import Any, Iterable, Iterator, List, NewType, Tuple, Union, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..types import ChunkKey

_INTEGER = (
    r"([1-9]+\d*|0)"  # matches 0 or an unsigned integer that does not begin with zero
)
_SEPARATOR = r"\."
_CHUNK_KEY = rf"^{_INTEGER}+({_SEPARATOR}{_INTEGER})*$"  # matches 1 integer, optionally followed by more integers each separated by a separator (i.e. a period)


ChunkDict = NewType("ChunkDict", dict[ChunkKey, dict[str, Union[str, int]]])


class ChunkEntry(BaseModel):
    """
    Information for a single chunk in the manifest.

    Stored in the form `{"path": "s3://bucket/foo.nc", "offset": 100, "length": 100}`.
    """

    model_config = ConfigDict(frozen=True)

    path: str  # TODO stricter typing/validation of possible local / remote paths?
    offset: int
    length: int

    def __repr__(self) -> str:
        return f"ChunkEntry(path='{self.path}', offset={self.offset}, length={self.length})"

    @classmethod
    def from_kerchunk(
        cls, path_and_byte_range_info: List[Union[str, int]]
    ) -> "ChunkEntry":
        path, offset, length = path_and_byte_range_info
        return ChunkEntry(path=path, offset=offset, length=length)

    def to_kerchunk(self) -> List[Union[str, int]]:
        """Write out in the format that kerchunk uses for chunk entries."""
        return [self.path, self.offset, self.length]

    def dict(self) -> dict[str, Union[str, int]]:
        return dict(path=self.path, offset=self.offset, length=self.length)


# TODO we want the path field to contain a variable-length string, but that's not available until numpy 2.0
# See https://numpy.org/neps/nep-0055-string_dtype.html
MANIFEST_STRUCTURED_ARRAY_DTYPES = np.dtype(
    [("path", "<U32"), ("offset", np.int32), ("length", np.int32)]
)


class ChunkManifest(BaseModel):
    """
    In-memory representation of a single Zarr chunk manifest.

    Stores the manifest internally as a numpy structured array.

    The manifest can be converted to or from a dictionary form looking like this


        {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }


    using the .from_dict() and .dict() methods, so users of this class can think of the manifest as if it were a dict.

    See the chunk manifest SPEC proposal in https://github.com/zarr-developers/zarr-specs/issues/287 .

    Validation is done when this object is instatiated, and this class is immutable,
    so it's not possible to have a ChunkManifest object that does not represent a complete valid grid of chunks.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,  # so pydantic doesn't complain about the numpy array field
    )

    # TODO how to type hint to indicate a numpy structured array with specifically-typed fields?
    entries: np.ndarray

    @classmethod
    def from_dict(cls, chunks: ChunkDict) -> "ChunkManifest":
        # TODO do some input validation here first?
        validate_chunk_keys(chunks.keys())

        # TODO should we actually pass shape in, in case there are not enough chunks to give correct idea of full shape?
        shape = get_chunk_grid_shape(chunks.keys())

        # Initializing to empty implies that entries with path='' are treated as missing chunks
        entries = np.empty(shape=shape, dtype=MANIFEST_STRUCTURED_ARRAY_DTYPES)

        # populate the array
        for key, entry in chunks.items():
            try:
                entries[split(key)] = tuple(entry.values())
            except (ValueError, TypeError) as e:
                msg = (
                    "Each chunk entry must be of the form dict(path=<str>, offset=<int>, length=<int>), "
                    f"but got {entry}"
                )
                raise ValueError(msg) from e

        return ChunkManifest(entries=entries)

    @property
    def ndim_chunk_grid(self) -> int:
        """
        Number of dimensions in the chunk grid.

        Not the same as the dimension of an array backed by this chunk manifest.
        """
        return self.entries.ndim

    @property
    def shape_chunk_grid(self) -> Tuple[int, ...]:
        """
        Number of separate chunks along each dimension.

        Not the same as the shape of an array backed by this chunk manifest.
        """
        return self.entries.shape

    def __repr__(self) -> str:
        return f"ChunkManifest<shape={self.shape_chunk_grid}>"

    def __getitem__(self, key: ChunkKey) -> ChunkEntry:
        indices = split(key)
        return ChunkEntry(self.entries[indices])

    def __iter__(self) -> Iterator[ChunkKey]:
        return iter(self.entries.keys())

    def __len__(self) -> int:
        return self.entries.size

    def dict(self) -> ChunkDict:
        """
        Converts the entire manifest to a nested dictionary, of the form

        {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        """

        def _entry_to_dict(entry: Tuple[str, int, int]) -> dict[str, Union[str, int]]:
            return {
                "path": entry[0],
                "offset": entry[1],
                "length": entry[2],
            }

        coord_vectors = np.mgrid[
            tuple(slice(None, length) for length in self.shape_chunk_grid)
        ]

        return cast(
            ChunkDict,
            {
                join(inds): _entry_to_dict(entry.item())
                for *inds, entry in np.nditer([*coord_vectors, self.entries])
                if entry.item()[0]
                != ""  # don't include entry if path='' (i.e. empty chunk)
            },
        )

    def __eq__(self, other: Any) -> bool:
        """Two manifests are equal if all of their entries are identical."""
        return (self.entries == other.entries).all()

    @classmethod
    def from_zarr_json(cls, filepath: str) -> "ChunkManifest":
        """Create a ChunkManifest from a Zarr manifest.json file."""
        with open(filepath, "r") as manifest_file:
            entries_dict = json.load(manifest_file)

        entries = {
            cast(ChunkKey, k): ChunkEntry(**entry) for k, entry in entries_dict.items()
        }
        return cls(entries=entries)

    def to_zarr_json(self, filepath: str) -> None:
        """Write a ChunkManifest to a Zarr manifest.json file."""
        with open(filepath, "w") as json_file:
            json.dump(self.dict(), json_file, indent=4, separators=(", ", ": "))

    @classmethod
    def _from_kerchunk_chunk_dict(cls, kerchunk_chunk_dict) -> "ChunkManifest":
        chunkentries = {
            cast(ChunkKey, k): ChunkEntry.from_kerchunk(v).dict()
            for k, v in kerchunk_chunk_dict.items()
        }
        return ChunkManifest.from_dict(cast(ChunkDict, chunkentries))


def split(key: ChunkKey) -> Tuple[int, ...]:
    return tuple(int(i) for i in key.split("."))


def join(inds: Iterable[Any]) -> ChunkKey:
    return cast(ChunkKey, ".".join(str(i) for i in list(inds)))


def get_ndim_from_key(key: str) -> int:
    """Get number of dimensions implied by key, e.g. '4.5.6' -> 3"""
    return len(key.split("."))


def validate_chunk_keys(chunk_keys: Iterable[ChunkKey]):
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


def get_chunk_grid_shape(chunk_keys: Iterable[ChunkKey]) -> Tuple[int, ...]:
    # find max chunk index along each dimension
    zipped_indices = zip(*[split(key) for key in chunk_keys])
    chunk_grid_shape = tuple(
        max(indices_along_one_dim) + 1 for indices_along_one_dim in zipped_indices
    )
    return chunk_grid_shape
