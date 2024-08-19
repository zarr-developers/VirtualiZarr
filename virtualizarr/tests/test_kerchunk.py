import numpy as np
import pandas as pd
import pytest
import ujson  # type: ignore
import xarray as xr
import xarray.testing as xrt

from virtualizarr.kerchunk import (
    FileType,
    _automatically_determine_filetype,
    find_var_names,
)
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.xarray import dataset_from_kerchunk_refs


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


def test_dataset_from_df_refs_with_filters():
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


class TestAccessor:
    def test_accessor_to_kerchunk_dict(self):
        manifest = ChunkManifest(
            entries={"0.0": dict(path="test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=np.nan,
                order="C",
            ),
        )
        ds = xr.Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"dtype":"<i8","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["test.nc", 6144, 48],
            },
        }

        result_ds_refs = ds.virtualize.to_kerchunk(format="dict")
        assert result_ds_refs == expected_ds_refs

    def test_accessor_to_kerchunk_json(self, tmp_path):
        manifest = ChunkManifest(
            entries={"0.0": dict(path="test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=np.nan,
                order="C",
            ),
        )
        ds = xr.Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"

        ds.virtualize.to_kerchunk(filepath, format="json")

        with open(filepath) as json_file:
            loaded_refs = ujson.load(json_file)

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"dtype":"<i8","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["test.nc", 6144, 48],
            },
        }
        assert loaded_refs == expected_ds_refs

    def test_accessor_to_kerchunk_parquet(self, tmp_path):
        chunks_dict = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 4),
                dtype=np.dtype("<i8"),
                chunks=(2, 2),
                compressor=None,
                filters=None,
                fill_value=None,
                order="C",
            ),
        )
        ds = xr.Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs"

        ds.virtualize.to_kerchunk(filepath, format="parquet", record_size=2)

        with open(tmp_path / "refs" / ".zmetadata") as f:
            meta = ujson.load(f)
            assert list(meta) == ["metadata", "record_size"]
            assert meta["record_size"] == 2

        df0 = pd.read_parquet(filepath / "a" / "refs.0.parq")

        assert df0.to_dict() == {
            "offset": {0: 100, 1: 200},
            "path": {
                0: "foo.nc",
                1: "foo.nc",
            },
            "size": {0: 100, 1: 100},
            "raw": {0: None, 1: None},
        }


def test_kerchunk_roundtrip_in_memory_no_concat():
    # Set up example xarray dataset
    chunks_dict = {
        "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(
        zarray=dict(
            shape=(2, 4),
            dtype=np.dtype("<i8"),
            chunks=(2, 2),
            compressor=None,
            filters=None,
            fill_value=np.nan,
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


def test_automatically_determine_filetype_netcdf3_netcdf4():
    # test the NetCDF3 vs NetCDF4 automatic file type selection

    ds = xr.Dataset({"a": (["x"], [0, 1])})
    netcdf3_file_path = "/tmp/netcdf3.nc"
    netcdf4_file_path = "/tmp/netcdf4.nc"

    # write two version of NetCDF
    ds.to_netcdf(netcdf3_file_path, engine="scipy", format="NETCDF3_CLASSIC")
    ds.to_netcdf(netcdf4_file_path, engine="h5netcdf")

    assert FileType("netcdf3") == _automatically_determine_filetype(
        filepath=netcdf3_file_path
    )
    assert FileType("hdf5") == _automatically_determine_filetype(
        filepath=netcdf4_file_path
    )


@pytest.mark.parametrize(
    "filetype,headerbytes",
    [
        ("netcdf3", b"CDF"),
        ("hdf5", b"\x89HDF"),
        ("grib", b"GRIB"),
        ("tiff", b"II*"),
        ("fits", b"SIMPLE"),
    ],
)
def test_valid_filetype_bytes(tmp_path, filetype, headerbytes):
    filepath = tmp_path / "file.abc"
    with open(filepath, "wb") as f:
        f.write(headerbytes)
    assert FileType(filetype) == _automatically_determine_filetype(filepath=filepath)


def test_notimplemented_filetype(tmp_path):
    for headerbytes in [b"JUNK", b"\x0e\x03\x13\x01"]:
        filepath = tmp_path / "file.abc"
        with open(filepath, "wb") as f:
            f.write(headerbytes)
        with pytest.raises(NotImplementedError):
            _automatically_determine_filetype(filepath=filepath)


def test_FileType():
    # tests if FileType converts user supplied strings to correct filetype
    assert "netcdf3" == FileType("netcdf3").name
    assert "netcdf4" == FileType("netcdf4").name
    assert "hdf4" == FileType("hdf4").name
    assert "hdf5" == FileType("hdf5").name
    assert "grib" == FileType("grib").name
    assert "tiff" == FileType("tiff").name
    assert "fits" == FileType("fits").name
    assert "zarr" == FileType("zarr").name
    with pytest.raises(ValueError):
        FileType(None)


def test_no_duplicates_find_var_names():
    """Verify that we get a deduplicated list of var names"""
    ref_dict = {"refs": {"x/something": {}, "x/otherthing": {}}}
    assert len(find_var_names(ref_dict)) == 1
