from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray import open_dataset
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset
from virtualizarr.backend import FileType
from virtualizarr.manifests import ManifestArray
from virtualizarr.readers.kerchunk import _automatically_determine_filetype
from virtualizarr.tests import has_astropy, has_tifffile, network, requires_s3fs


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


class TestOpenVirtualDatasetIndexes:
    def test_no_indexes(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert vds.indexes == {}

    def test_create_default_indexes(self, netcdf4_file):
        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds = open_virtual_dataset(netcdf4_file, indexes=None)
        ds = open_dataset(netcdf4_file, decode_times=False)

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


class TestDetermineCoords:
    def test_determine_all_coords(self, netcdf4_file_with_2d_coords):
        vds = open_virtual_dataset(netcdf4_file_with_2d_coords, indexes={})

        expected_dimension_coords = ["ocean_time", "s_rho"]
        expected_2d_coords = ["lon_rho", "lat_rho", "h"]
        expected_1d_non_dimension_coords = ["Cs_r"]
        expected_scalar_coords = ["hc", "Vtransform"]
        expected_coords = (
            expected_dimension_coords
            + expected_2d_coords
            + expected_1d_non_dimension_coords
            + expected_scalar_coords
        )
        assert set(vds.coords) == set(expected_coords)

        # print(vds.attrs)
        # assert False

        # TODO assert coord attributes have been altered
        for coord_name in expected_coords:
            print(vds[coord_name].attrs)
            # assert vds[coord_name].attrs['']

        # assert False


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
        vds = open_virtual_dataset(
            fpath,
            filetype=filetype,
            indexes=indexes,
            reader_options={"storage_options": {"anon": True}},
        )

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
            pytest.param(
                "fits",
                "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits",
                marks=pytest.mark.skipif(
                    not has_astropy, reason="package astropy is not available"
                ),
            ),
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
            vds = open_virtual_dataset(url, indexes={})
            assert isinstance(vds, xr.Dataset)


class TestLoadVirtualDataset:
    def test_loadable_variables(self, netcdf4_file):
        vars_to_load = ["air", "time"]
        vds = open_virtual_dataset(
            netcdf4_file, loadable_variables=vars_to_load, indexes={}
        )

        for name in vds.variables:
            if name in vars_to_load:
                assert isinstance(vds[name].data, np.ndarray), name
            else:
                assert isinstance(vds[name].data, ManifestArray), name

        full_ds = xr.open_dataset(netcdf4_file, decode_times=False)

        for name in full_ds.variables:
            if name in vars_to_load:
                xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    def test_explicit_filetype(self, netcdf4_file):
        with pytest.raises(ValueError):
            open_virtual_dataset(netcdf4_file, filetype="unknown")

        with pytest.raises(NotImplementedError):
            open_virtual_dataset(netcdf4_file, filetype="grib")

    @patch("virtualizarr.readers.kerchunk.read_kerchunk_references_from_file")
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

    def test_open_dataset_with_empty(self, hdf5_empty, tmpdir):
        vds = open_virtual_dataset(hdf5_empty)
        assert vds.empty.dims == ()
        assert vds.empty.attrs == {"empty": "true"}

    def test_open_dataset_with_scalar(self, hdf5_scalar, tmpdir):
        vds = open_virtual_dataset(hdf5_scalar)
        assert vds.scalar.dims == ()
        assert vds.scalar.attrs == {"scalar": "true"}


def test_cftime_variables_must_be_in_loadable_variables(tmpdir):
    ds = xr.Dataset(data_vars={"time": ["2024-06-21"]})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")
    with pytest.raises(ValueError, match="'time' not in"):
        open_virtual_dataset(f"{tmpdir}/scalar.nc", cftime_variables=["time"])
