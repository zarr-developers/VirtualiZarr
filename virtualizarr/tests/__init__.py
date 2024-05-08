import numpy as np

from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray
from virtualizarr.zarr import ZArray


def create_manifestarray(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> ManifestArray:
    """
    Create an example ManifestArray with sensible defaults.
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

    if shape != ():
        raise NotImplementedError(
            "Only generation of array representing a single scalar currently supported"
        )

    # TODO generalize this
    chunkmanifest = ChunkManifest(
        entries={"0": ChunkEntry(path="scalar.nc", offset=6144, length=48)}
    )

    return ManifestArray(chunkmanifest=chunkmanifest, zarray=zarray)
