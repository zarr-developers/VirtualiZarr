import numpy as np
import xarray as xr
import xarray.testing as xrt

from virtualizarr import ManifestArray, open_virtual_dataset
from virtualizarr.manifests.manifest import ChunkManifest


def test_zarr_v3_roundtrip(tmpdir):
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="test.nc", offset=6144, length=48)}
        ),
        zarray=dict(
            shape=(2, 3),
            data_type=np.dtype("<i8"),
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": [2, 3]}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "."}},
            codecs=(),
            attributes={},
            dimension_names=None,
            fill_value=np.nan,
            zarr_format=3,
        ),
    )
    original = xr.Dataset({"a": (["x", "y"], arr)}, attrs={"something": 0})

    original.virtualize.to_zarr(tmpdir / "store.zarr")
    roundtrip = open_virtual_dataset(
        tmpdir / "store.zarr", filetype="zarr_v3", indexes={}
    )

    xrt.assert_identical(roundtrip, original)
