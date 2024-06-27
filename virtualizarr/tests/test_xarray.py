from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import has_tifffile, network, requires_s3fs
from virtualizarr.zarr import ZArray


def test_wrapping():
    chunks = (5, 10)
    shape = (5, 20)
    dtype = np.dtype("int32")
    zarray = ZArray(
        chunks=chunks,
        compressor="zlib",
        dtype=dtype,
        fill_value=0.0,
        filters=None,
        order="C",
        shape=shape,
        zarr_format=2,
    )

    chunks_dict = {
        "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(zarray=zarray, chunkmanifest=manifest)
    ds = xr.Dataset({"a": (["x", "y"], marr)})

    assert isinstance(ds["a"].data, ManifestArray)
    assert ds["a"].shape == shape
    assert ds["a"].dtype == dtype
    assert ds["a"].chunks == chunks


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


# TODO refactor these tests by making some fixtures
class TestConcat:
    def test_concat_along_existing_dim(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="x")["a"]
        assert result.indexes == {}

        assert result.shape == (2, 20)
        assert result.chunks == (1, 10)
        assert result.data.manifest.dict() == {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format

    def test_concat_along_new_dim(self):
        # this calls np.stack internally
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="z")["a"]
        assert result.indexes == {}

        # xarray.concat adds new dimensions along axis=0
        assert result.shape == (2, 5, 20)
        assert result.chunks == (1, 5, 10)
        assert result.data.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "1.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format

    def test_concat_dim_coords_along_existing_dim(self):
        # Tests that dimension coordinates don't automatically get new indexes on concat
        # See https://github.com/pydata/xarray/issues/8871

        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(10,),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(20,),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0": {"path": "foo.nc", "offset": 100, "length": 100},
            "1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        coords = xr.Coordinates({"t": (["t"], marr1)}, indexes={})
        ds1 = xr.Dataset(coords=coords)

        chunks_dict2 = {
            "0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        coords = xr.Coordinates({"t": (["t"], marr2)}, indexes={})
        ds2 = xr.Dataset(coords=coords)

        result = xr.concat([ds1, ds2], dim="t")["t"]
        assert result.indexes == {}

        assert result.shape == (40,)
        assert result.chunks == (10,)
        assert result.data.manifest.dict() == {
            "0": {"path": "foo.nc", "offset": 100, "length": 100},
            "1": {"path": "foo.nc", "offset": 200, "length": 100},
            "2": {"path": "foo.nc", "offset": 300, "length": 100},
            "3": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format


class TestOpenVirtualDatasetAttrs:
    def test_drop_array_dimensions(self, netcdf4_file):
        # regression test for GH issue #150
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert "_ARRAY_DIMENSIONS" not in vds["air"].attrs

    def test_coordinate_variable_attrs_preserved(self, netcdf4_file):
        # regression test for GH issue #155
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert vds["lat"].attrs == {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        }


class TestOpenVirtualDatasetIndexes:
    def test_no_indexes(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert vds.indexes == {}

    def test_create_default_indexes(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes=None)
        ds = xr.open_dataset(netcdf4_file)

        # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
        assert index_mappings_equal(vds.xindexes, ds.xindexes)


def index_mappings_equal(indexes1: Mapping[str, Index], indexes2: Mapping[str, Index]):
    # Check if the mappings have the same keys
    if set(indexes1.keys()) != set(indexes2.keys()):
        return False

    # Check if the values for each key are identical
    for key in indexes1.keys():
        index1 = indexes1[key]
        index2 = indexes2[key]

        if not index1.equals(index2):
            return False

    return True


class TestCombineUsingIndexes:
    def test_combine_by_coords(self, netcdf4_files):
        filepath1, filepath2 = netcdf4_files

        vds1 = open_virtual_dataset(filepath1)
        vds2 = open_virtual_dataset(filepath2)

        combined_vds = xr.combine_by_coords(
            [vds2, vds1],
        )

        assert combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing


@network
@requires_s3fs
class TestReadFromS3:
    @pytest.mark.parametrize(
        "filetype", ["netcdf4", None], ids=["netcdf4 filetype", "None filetype"]
    )
    @pytest.mark.parametrize(
        "indexes", [None, {}], ids=["None index", "empty dict index"]
    )
    def test_anon_read_s3(self, filetype, indexes):
        """Parameterized tests for empty vs supplied indexes and filetypes."""
        # TODO: Switch away from this s3 url after minIO is implemented.
        fpath = "s3://carbonplan-share/virtualizarr/local.nc"
        vds = open_virtual_dataset(fpath, filetype=filetype, indexes=indexes)

        assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}
        for var in vds.variables:
            assert isinstance(vds[var].data, ManifestArray), var


@network
class TestReadFromURL:
    @pytest.mark.parametrize(
        "filetype, url",
        [
            (
                "grib",
                "https://github.com/pydata/xarray-data/raw/master/era5-2mt-2019-03-uk.grib",
            ),
            (
                "netcdf3",
                "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
            ),
            (
                "netcdf4",
                "https://github.com/pydata/xarray-data/raw/master/ROMS_example.nc",
            ),
            (
                "hdf4",
                "https://github.com/corteva/rioxarray/raw/master/test/test_data/input/MOD09GA.A2008296.h14v17.006.2015181011753.hdf",
            ),
            # https://github.com/zarr-developers/VirtualiZarr/issues/159
            # ("hdf5", "https://github.com/fsspec/kerchunk/raw/main/kerchunk/tests/NEONDSTowerTemperatureData.hdf5"),
            pytest.param(
                "tiff",
                "https://github.com/fsspec/kerchunk/raw/main/kerchunk/tests/lcmap_tiny_cog_2020.tif",
                marks=pytest.mark.skipif(
                    not has_tifffile, reason="package tifffile is not available"
                ),
            ),
            # ("fits", "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"),
            (
                "jpg",
                "https://github.com/rasterio/rasterio/raw/main/tests/data/389225main_sw_1965_1024.jpg",
            ),
        ],
    )
    def test_read_from_url(self, filetype, url):
        if filetype in ["grib", "jpg", "hdf4"]:
            with pytest.raises(NotImplementedError):
                vds = open_virtual_dataset(url, reader_options={}, indexes={})
        else:
            vds = open_virtual_dataset(url, reader_options={}, indexes={})
            assert isinstance(vds, xr.Dataset)


class TestLoadVirtualDataset:
    def test_loadable_variables(self, netcdf4_file):
        vars_to_load = ["air", "time"]
        vds = open_virtual_dataset(netcdf4_file, loadable_variables=vars_to_load)

        for name in vds.variables:
            if name in vars_to_load:
                assert isinstance(vds[name].data, np.ndarray), name
            else:
                assert isinstance(vds[name].data, ManifestArray), name

        full_ds = xr.open_dataset(netcdf4_file)

        for name in full_ds.variables:
            if name in vars_to_load:
                xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    def test_explicit_filetype(self, netcdf4_file):
        with pytest.raises(ValueError):
            open_virtual_dataset(netcdf4_file, filetype="unknown")

        with pytest.raises(NotImplementedError):
            open_virtual_dataset(netcdf4_file, filetype="grib")

    @patch("virtualizarr.kerchunk.read_kerchunk_references_from_file")
    def test_open_virtual_dataset_passes_expected_args(
        self, mock_read_kerchunk, netcdf4_file
    ):
        reader_options = {"option1": "value1", "option2": "value2"}
        open_virtual_dataset(netcdf4_file, indexes={}, reader_options=reader_options)
        args = {
            "filepath": netcdf4_file,
            "filetype": None,
            "reader_options": reader_options,
        }
        mock_read_kerchunk.assert_called_once_with(**args)


class TestRenamePaths:
    def test_rename_to_str(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )

    def test_rename_using_function(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})

        def local_to_s3_url(old_local_path: str) -> str:
            from pathlib import Path

            new_s3_bucket_url = "s3://bucket/"

            filename = Path(old_local_path).name
            return str(new_s3_bucket_url + filename)

        renamed_vds = vds.virtualize.rename_paths(local_to_s3_url)
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )

    def test_invalid_type(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})

        with pytest.raises(TypeError):
            vds.virtualize.rename_paths(["file1.nc", "file2.nc"])

    def test_mixture_of_manifestarrays_and_numpy_arrays(self, netcdf4_file):
        vds = open_virtual_dataset(
            netcdf4_file, indexes={}, loadable_variables=["lat", "lon"]
        )
        renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )
        assert isinstance(renamed_vds["lat"].data, np.ndarray)
