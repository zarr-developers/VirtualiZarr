import itertools

import numpy as np

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.zarr import ZArray, ceildiv


def create_manifestarray(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> ManifestArray:
    """
    Create an example ManifestArray with sensible defaults.

    The manifest is populated with a (somewhat) unique path, offset, and length for each key.
    """

    zarray = ZArray(
        chunks=chunks,
        compressor="zlib",
        dtype=np.dtype("float32"),
        fill_value=0.0,  # TODO change this to NaN?
        filters=None,
        order="C",
        shape=shape,
        zarr_format=2,
    )

    chunk_grid_shape = tuple(
        ceildiv(axis_length, chunk_length)
        for axis_length, chunk_length in zip(shape, chunks)
    )

    if chunk_grid_shape == ():
        d = {"0": entry_from_chunk_key((0,))}
    else:
        # create every possible combination of keys
        all_possible_combos = itertools.product(
            *[range(length) for length in chunk_grid_shape]
        )
        d = {join(ind): entry_from_chunk_key(ind) for ind in all_possible_combos}

    chunkmanifest = ChunkManifest(entries=d)

    return ManifestArray(chunkmanifest=chunkmanifest, zarray=zarray)


def entry_from_chunk_key(ind: tuple[int, ...]) -> dict[str, str | int]:
    """Generate a (somewhat) unique manifest entry from a given chunk key"""
    entry = {
        "path": f"file.{str(join(ind))}.nc",
        "offset": offset_from_chunk_key(ind),
        "length": length_from_chunk_key(ind),
    }
    return entry  # type: ignore[return-value]


def offset_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) * 10


def length_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) + 5
