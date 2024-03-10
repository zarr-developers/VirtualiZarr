import itertools
import re
from typing import Any, Iterable, Iterator, List, Mapping, Tuple, Union, cast

from pydantic import BaseModel, ConfigDict, validator

from ..types import ChunkKey

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
