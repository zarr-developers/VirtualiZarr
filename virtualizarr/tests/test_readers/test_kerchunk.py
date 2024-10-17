import numpy as np

from virtualizarr.manifests import ManifestArray
from virtualizarr.readers.kerchunk import (
    dataset_from_kerchunk_refs,
)
from virtualizarr.tests import requires_kerchunk


def gen_ds_refs(
    zgroup: str = '{"zarr_format":2}',
    zarray: str = '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
    zattrs: str = '{"_ARRAY_DIMENSIONS":["x","y"]}',
    chunk: list = ["test1.nc", 6144, 48],
):
    return {
        "version": 1,
        "refs": {
            ".zgroup": zgroup,
            "a/.zarray": zarray,
            "a/.zattrs": zattrs,
            "a/0.0": chunk,
        },
    }


@requires_kerchunk
def test_dataset_from_df_refs():
    ds_refs = gen_ds_refs()
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
    assert da.data.zarray.fill_value is np.nan
    assert da.data.zarray.order == "C"

    assert da.data.manifest.dict() == {
        "0.0": {"path": "test1.nc", "offset": 6144, "length": 48}
    }


@requires_kerchunk
def test_dataset_from_df_refs_with_filters():
    import ujson

    filters = [{"elementsize": 4, "id": "shuffle"}, {"id": "zlib", "level": 4}]
    zarray = {
        "chunks": [2, 3],
        "compressor": None,
        "dtype": "<i8",
        "fill_value": None,
        "filters": filters,
        "order": "C",
        "shape": [2, 3],
        "zarr_format": 2,
    }
    ds_refs = gen_ds_refs(zarray=ujson.dumps(zarray))
    ds = dataset_from_kerchunk_refs(ds_refs)
    da = ds["a"]
    assert da.data.zarray.filters == filters
