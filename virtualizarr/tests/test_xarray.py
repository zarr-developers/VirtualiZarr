import numpy as np

from virtualizarr.manifests import ManifestArray
from virtualizarr.xarray import dataset_from_kerchunk_refs


def test_dataset_from_kerchunk_refs():
    ds_refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            "a/.zarray": '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": ["test1.nc", 6144, 48],
        },
    }

    ds = dataset_from_kerchunk_refs(ds_refs)

    assert "a" in ds
    da = ds["a"]
    assert isinstance(da.data, ManifestArray)
    assert da.dims == ("x", "y")
    assert da.shape == (2, 3)
    assert da.chunks == (2, 3)
    assert da.dtype == np.dtype("<i8")

    assert da.data.zarray.compressor is None
    assert da.data.zarray.filters is None
    assert da.data.zarray.fill_value is None
    assert da.data.zarray.order == "C"

    assert da.data.manifest.dict() == {
        "0.0": {"path": "test1.nc", "offset": 6144, "length": 48}
    }
