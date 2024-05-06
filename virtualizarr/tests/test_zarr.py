import numpy as np
import xarray as xr
import xarray.testing as xrt

from virtualizarr import ManifestArray, open_virtual_dataset
from virtualizarr.manifests.manifest import ChunkEntry


def test_zarr_v3_roundtrip(tmpdir):
    arr = ManifestArray(
        chunkmanifest={"0.0": ChunkEntry(path="test.nc", offset=6144, length=48)},
        zarray=dict(
            shape=(2, 3),
            dtype=np.dtype("<i8"),
            chunks=(2, 3),
            compressor=None,
            filters=None,
            fill_value=None,
            order="C",
            zarr_format=3,
        ),
    )
    original = xr.Dataset({"a": (["x", "y"], arr)}, attrs={"something": 0})

    original.virtualize.to_zarr(tmpdir / "store.zarr")
    roundtrip = open_virtual_dataset(
        tmpdir / "store.zarr", filetype="zarr_v3", indexes={}
    )

    xrt.assert_identical(roundtrip, original)
