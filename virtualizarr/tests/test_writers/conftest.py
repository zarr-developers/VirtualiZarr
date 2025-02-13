import numpy as np
import pytest
from xarray import Dataset
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray


@pytest.fixture
def vds_with_manifest_arrays(
    array_v3_metadata, arraybytes_codec, zlib_codec
) -> Dataset:
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="/test.nc", offset=6144, length=48)}
        ),
        metadata=array_v3_metadata(
            shape=(2, 3),
            data_type=np.dtype("<i8"),
            chunks=(2, 3),
            codecs=[
                arraybytes_codec,
                zlib_codec,
            ],
            fill_value=0,
        ),
    )
    var = Variable(dims=["x", "y"], data=arr, attrs={"units": "km"})
    return Dataset({"a": var}, attrs={"something": 0})
