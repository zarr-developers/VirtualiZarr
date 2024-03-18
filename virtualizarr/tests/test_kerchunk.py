import numpy as np
import ujson  # type: ignore
import xarray as xr
import xarray.testing as xrt

from virtualizarr.manifests import ChunkManifest, ManifestArray
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


class TestAccessor:
    def test_accessor_to_kerchunk_dict(self):
        manifest = ChunkManifest.from_dict(
            {"0.0": dict(path="test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=None,
                order="C",
            ),
        )
        ds = xr.Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                "a/.zarray": '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["test.nc", 6144, 48],
            },
        }

        result_ds_refs = ds.virtualize.to_kerchunk(format="dict")
        assert result_ds_refs == expected_ds_refs

    def test_accessor_to_kerchunk_json(self, tmp_path):
        manifest = ChunkManifest.from_dict(
            {"0.0": dict(path="test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=None,
                order="C",
            ),
        )
        ds = xr.Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"

        ds.virtualize.to_kerchunk(filepath, format="json")

        with open(filepath, "r") as json_file:
            loaded_refs = ujson.load(json_file)

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                "a/.zarray": '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["test.nc", 6144, 48],
            },
        }
        assert loaded_refs == expected_ds_refs


def test_kerchunk_roundtrip_in_memory_no_concat():
    # Set up example xarray dataset
    chunks_dict = {
        "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest.from_dict(chunks_dict)
    marr = ManifestArray(
        zarray=dict(
            shape=(2, 4),
            dtype=np.dtype("<i8"),
            chunks=(2, 2),
            compressor=None,
            filters=None,
            fill_value=None,
            order="C",
        ),
        chunkmanifest=manifest,
    )
    ds = xr.Dataset({"a": (["x", "y"], marr)})

    # Use accessor to write it out to kerchunk reference dict
    ds_refs = ds.virtualize.to_kerchunk(format="dict")

    # Use dataset_from_kerchunk_refs to reconstruct the dataset
    roundtrip = dataset_from_kerchunk_refs(ds_refs)

    # Assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)
