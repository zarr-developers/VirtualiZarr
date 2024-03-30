import itertools
import json
import re
from typing import Any, Iterable, Iterator, List, Mapping, Tuple, Union, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

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

    @field_validator("entries")
    @classmethod
    def validate_chunks(cls, entries: Any) -> Mapping[ChunkKey, ChunkEntry]:
        validate_chunk_keys(list(entries.keys()))

        # TODO what if pydantic adjusts anything during validation?
        return entries

    @property
    def ndim_chunk_grid(self) -> int:
        """
        Number of dimensions in the chunk grid.

        Not the same as the dimension of an array backed by this chunk manifest.
        """
        return get_ndim_from_key(list(self.entries.keys())[0])

    @property
    def shape_chunk_grid(self) -> Tuple[int, ...]:
        """
        Number of separate chunks along each dimension.

        Not the same as the shape of an array backed by this chunk manifest.
        """
        return get_chunk_grid_shape(list(self.entries.keys()))

    def __repr__(self) -> str:
        return f"ChunkManifest<shape={self.shape_chunk_grid}>"

    def __getitem__(self, key: ChunkKey) -> ChunkEntry:
        return self.chunks[key]

    def __iter__(self) -> Iterator[ChunkKey]:
        return iter(self.chunks.keys())

    def __len__(self) -> int:
        return len(self.chunks)

    def dict(self) -> dict[str, dict[str, Union[str, int]]]:
        """Converts the entire manifest to a nested dictionary."""
        return {k: dict(entry) for k, entry in self.entries.items()}

    @staticmethod
    def from_zarr_json(filepath: str) -> "ChunkManifest":
        """Create a ChunkManifest from a Zarr manifest.json file."""
        raise NotImplementedError()

    def to_zarr_json(self, filepath: str) -> None:
        """Write a ChunkManifest to a Zarr manifest.json file."""
        with open(filepath, "w") as json_file:
            json.dump(self.dict(), json_file, indent=4, separators=(", ", ": "))

    @classmethod
    def _from_kerchunk_chunk_dict(cls, kerchunk_chunk_dict) -> "ChunkManifest":
        chunkentries = {
            k: ChunkEntry.from_kerchunk(v) for k, v in kerchunk_chunk_dict.items()
        }
        return ChunkManifest(entries=chunkentries)


def split(key: ChunkKey) -> List[int]:
    return list(int(i) for i in key.split("."))


def join(inds: Iterable[int]) -> ChunkKey:
    return cast(ChunkKey, ".".join(str(i) for i in inds))


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

    # Check that the keys collectively form a complete grid
    check_keys_form_grid(chunk_keys)


def get_chunk_grid_shape(chunk_keys: Iterable[ChunkKey]) -> Tuple[int, ...]:
    # find max chunk index along each dimension
    zipped_indices = zip(*[split(key) for key in chunk_keys])
    chunk_grid_shape = tuple(
        max(indices_along_one_dim) + 1 for indices_along_one_dim in zipped_indices
    )
    return chunk_grid_shape


def check_keys_form_grid(chunk_keys: Iterable[ChunkKey]):
    """Check that the chunk keys collectively form a complete grid"""

    chunk_grid_shape = get_chunk_grid_shape(chunk_keys)

    # create every possible combination
    all_possible_combos = itertools.product(
        *[range(length) for length in chunk_grid_shape]
    )
    all_required_chunk_keys: set[ChunkKey] = set(
        join(inds) for inds in all_possible_combos
    )

    # check that every possible combination is represented once in the list of chunk keys
    if set(chunk_keys) != all_required_chunk_keys:
        raise ValueError("Chunk keys do not form a complete grid")


def concat_manifests(manifests: List["ChunkManifest"], axis: int) -> "ChunkManifest":
    """
    Concatenate manifests along an existing dimension.

    This only requires adjusting one index of chunk keys along a single dimension.

    Note axis is not expected to be negative.
    """
    if len(manifests) == 1:
        return manifests[0]

    chunk_grid_shapes = [manifest.shape_chunk_grid for manifest in manifests]
    lengths_along_concat_dim = [shape[axis] for shape in chunk_grid_shapes]

    # Note we do not need to change the keys of the first manifest
    chunk_index_offsets = np.cumsum(lengths_along_concat_dim)[:-1]
    new_entries = [
        adjust_chunk_keys(manifest.entries, axis, offset)
        for manifest, offset in zip(manifests[1:], chunk_index_offsets)
    ]
    all_entries = [manifests[0].entries] + new_entries
    merged_entries = dict((k, v) for d in all_entries for k, v in d.items())

    # Arguably don't need to re-perform validation checks on a manifest we created out of already-validated manifests
    # Could use pydantic's model_construct classmethod to skip these checks
    # But we should actually performance test it because it might be pointless, and current implementation is safer
    return ChunkManifest(entries=merged_entries)


def adjust_chunk_keys(
    entries: Mapping[ChunkKey, ChunkEntry], axis: int, offset: int
) -> Mapping[ChunkKey, ChunkEntry]:
    """Replace all chunk keys with keys which have been offset along one axis."""

    def offset_key(key: ChunkKey, axis: int, offset: int) -> ChunkKey:
        inds = split(key)
        inds[axis] += offset
        return join(inds)

    return {offset_key(k, axis, offset): v for k, v in entries.items()}


def stack_manifests(manifests: List[ChunkManifest], axis: int) -> "ChunkManifest":
    """
    Stack manifests along a new dimension.

    This only requires inserting one index into all chunk keys to add a new dimension.

    Note axis is not expected to be negative.
    """

    # even if there is only one manifest it still needs a new axis inserted
    chunk_indexes_along_new_dim = range(len(manifests))
    new_entries = [
        insert_new_axis_into_chunk_keys(manifest.entries, axis, new_index_value)
        for manifest, new_index_value in zip(manifests, chunk_indexes_along_new_dim)
    ]
    merged_entries = dict((k, v) for d in new_entries for k, v in d.items())

    # Arguably don't need to re-perform validation checks on a manifest we created out of already-validated manifests
    # Could use pydantic's model_construct classmethod to skip these checks
    # But we should actually performance test it because it might be pointless, and current implementation is safer
    return ChunkManifest(entries=merged_entries)


def insert_new_axis_into_chunk_keys(
    entries: Mapping[ChunkKey, ChunkEntry], axis: int, new_index_value: int
) -> Mapping[ChunkKey, ChunkEntry]:
    """Replace all chunk keys with keys which have a new axis inserted, with a given value."""

    def insert_axis(key: ChunkKey, new_axis: int, index_value: int) -> ChunkKey:
        inds = split(key)
        inds.insert(new_axis, index_value)
        return join(inds)

    return {insert_axis(k, axis, new_index_value): v for k, v in entries.items()}
