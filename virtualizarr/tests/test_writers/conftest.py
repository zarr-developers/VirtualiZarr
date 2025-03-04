import numpy as np
import pytest
from xarray import Dataset
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray


@pytest.fixture
def vds_with_manifest_arrays() -> Dataset:
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="/test.nc", offset=6144, length=48)}
        ),
        zarray=dict(
            shape=(2, 3),
            dtype=np.dtype("<i8"),
            chunks=(2, 3),
            compressor={"id": "zlib", "level": 1},
            filters=None,
            fill_value=0,
            order="C",
            zarr_format=3,
        ),
    )
    var = Variable(dims=["x", "y"], data=arr, attrs={"units": "km"})
    return Dataset({"a": var}, attrs={"something": 0})
