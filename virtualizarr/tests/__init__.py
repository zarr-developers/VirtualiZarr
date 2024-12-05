import importlib
import itertools

import numpy as np
import pytest
from packaging.version import Version

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.zarr import ZArray, ceildiv

network = pytest.mark.network


def _importorskip(
    modname: str, minversion: str | None = None
) -> tuple[bool, pytest.MarkDecorator]:
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            v = getattr(mod, "__version__", "999")
            if Version(v) < Version(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False

    reason = f"requires {modname}"
    if minversion is not None:
        reason += f">={minversion}"
    func = pytest.mark.skipif(not has, reason=reason)
    return has, func


has_astropy, requires_astropy = _importorskip("astropy")
has_kerchunk, requires_kerchunk = _importorskip("kerchunk")
has_s3fs, requires_s3fs = _importorskip("s3fs")
has_scipy, requires_scipy = _importorskip("scipy")
has_tifffile, requires_tifffile = _importorskip("tifffile")
has_imagecodecs, requires_imagecodecs = _importorskip("imagecodecs")
has_hdf5plugin, requires_hdf5plugin = _importorskip("hdf5plugin")
has_zarr_python, requires_zarr_python = _importorskip("zarr")
has_zarr_python_v3, requires_zarr_python_v3 = _importorskip("zarr", "3.0.0b")


def create_manifestarray(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> ManifestArray:
    """
    Create an example ManifestArray with sensible defaults.

    The manifest is populated with a (somewhat) unique path, offset, and length for each key.
    """

    zarray = ZArray(
        chunks=chunks,
        compressor={"id": "blosc", "clevel": 5, "cname": "lz4", "shuffle": 1},
        dtype=np.dtype("float32"),
        fill_value=0.0,
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
        "path": f"/foo.{str(join(ind))}.nc",
        "offset": offset_from_chunk_key(ind),
        "length": length_from_chunk_key(ind),
    }
    return entry  # type: ignore[return-value]


def offset_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) * 10


def length_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) + 5
