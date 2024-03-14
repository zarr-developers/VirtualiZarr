import numpy as np
import xarray as xr

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.zarr import ZArray


class TestEquals:
    # regression test for GH29 https://github.com/TomNicholas/VirtualiZarr/issues/29
    def test_equals(self):
        chunks = (5, 10)
        shape = (5, 20)
        zarray = ZArray(
            chunks=chunks,
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})
        assert ds1.equals(ds2)

        chunks_dict3 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest3 = ChunkManifest(entries=chunks_dict3)
        marr3 = ManifestArray(zarray=zarray, chunkmanifest=manifest3)
        ds3 = xr.Dataset({"a": (["x", "y"], marr3)})
        assert not ds1.equals(ds3)
