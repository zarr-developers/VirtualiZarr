from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray import open_dataset
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset
from virtualizarr.backend import FileType, automatically_determine_filetype
from virtualizarr.manifests import ManifestArray
from virtualizarr.readers import HDF5VirtualBackend
from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.tests import (
    has_astropy,
    network,
    requires_kerchunk,
    requires_s3fs,
    requires_scipy,
)


@requires_scipy
def test_automatically_determine_filetype_netcdf3_netcdf4():
    # test the NetCDF3 vs NetCDF4 automatic file type selection

    ds = xr.Dataset({"a": (["x"], [0, 1])})
    netcdf3_file_path = "/tmp/netcdf3.nc"
    netcdf4_file_path = "/tmp/netcdf4.nc"

    # write two version of NetCDF
    ds.to_netcdf(netcdf3_file_path, engine="scipy", format="NETCDF3_CLASSIC")
    ds.to_netcdf(netcdf4_file_path, engine="h5netcdf")

    assert FileType("netcdf3") == automatically_determine_filetype(
        filepath=netcdf3_file_path
    )
    assert FileType("hdf5") == automatically_determine_filetype(
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
    assert FileType(filetype) == automatically_determine_filetype(filepath=filepath)


def test_notimplemented_filetype(tmp_path):
    for headerbytes in [b"JUNK", b"\x0e\x03\x13\x01"]:
        filepath = tmp_path / "file.abc"
        with open(filepath, "wb") as f:
            f.write(headerbytes)
        with pytest.raises(NotImplementedError):
            automatically_determine_filetype(filepath=filepath)


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


@requires_kerchunk
@pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
class TestOpenVirtualDatasetIndexes:
    def test_no_indexes(self, netcdf4_file, hdf_backend):
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)
        assert vds.indexes == {}

    def test_create_default_indexes(self, netcdf4_file, hdf_backend):
        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds = open_virtual_dataset(netcdf4_file, indexes=None, backend=hdf_backend)
        ds = open_dataset(netcdf4_file, decode_times=True)

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


@requires_kerchunk
@pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
def test_cftime_index(tmpdir, hdf_backend):
    """Ensure a virtual dataset contains the same indexes as an Xarray dataset"""
    # Note: Test was created to debug: https://github.com/zarr-developers/VirtualiZarr/issues/168
    ds = xr.Dataset(
        data_vars={
            "tasmax": (["time", "lat", "lon"], np.random.rand(2, 18, 36)),
        },
        coords={
            "time": np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]"),
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
        },
        attrs={"attr1_key": "attr1_val"},
    )
    ds.to_netcdf(f"{tmpdir}/tmp.nc")
    vds = open_virtual_dataset(
        f"{tmpdir}/tmp.nc",
        loadable_variables=["time", "lat", "lon"],
        indexes={},
        backend=hdf_backend,
    )
    # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
    assert index_mappings_equal(vds.xindexes, ds.xindexes)
    assert list(ds.coords) == list(vds.coords)
    assert vds.dims == ds.dims
    assert vds.attrs == ds.attrs


@requires_kerchunk
@pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
class TestOpenVirtualDatasetAttrs:
    def test_drop_array_dimensions(self, netcdf4_file, hdf_backend):
        # regression test for GH issue #150
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)
        assert "_ARRAY_DIMENSIONS" not in vds["air"].attrs

    def test_coordinate_variable_attrs_preserved(self, netcdf4_file, hdf_backend):
        # regression test for GH issue #155
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)
        assert vds["lat"].attrs == {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        }


@requires_kerchunk
class TestDetermineCoords:
    def test_infer_one_dimensional_coords(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert set(vds.coords) == {"time", "lat", "lon"}

    def test_var_attr_coords(self, netcdf4_file_with_2d_coords):
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


@network
@requires_s3fs
class TestReadFromS3:
    @pytest.mark.parametrize(
        "indexes", [None, {}], ids=["None index", "empty dict index"]
    )
    @pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
    def test_anon_read_s3(self, indexes, hdf_backend):
        """Parameterized tests for empty vs supplied indexes and filetypes."""
        # TODO: Switch away from this s3 url after minIO is implemented.
        fpath = "s3://carbonplan-share/virtualizarr/local.nc"
        vds = open_virtual_dataset(
            fpath,
            indexes=indexes,
            reader_options={"storage_options": {"anon": True}},
            backend=hdf_backend,
        )

        assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}
        for var in vds.variables:
            assert isinstance(vds[var].data, ManifestArray), var


@network
@pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
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
            (
                "hdf5",
                "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/GCOV/ALOS1_Rosamond_20081012/NISAR_L2_PR_GCOV_001_005_A_219_4020_SHNA_A_20081012T060910_20081012T060926_P01101_F_N_J_001.h5",
            ),
            # https://github.com/zarr-developers/VirtualiZarr/issues/159
            # ("hdf5", "https://github.com/fsspec/kerchunk/raw/main/kerchunk/tests/NEONDSTowerTemperatureData.hdf5"),
            pytest.param(
                "tiff",
                "https://github.com/fsspec/kerchunk/raw/main/kerchunk/tests/lcmap_tiny_cog_2020.tif",
                marks=pytest.mark.xfail(reason="not yet implemented"),
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
    def test_read_from_url(self, hdf_backend, filetype, url):
        if filetype in ["grib", "jpg", "hdf4"]:
            with pytest.raises(NotImplementedError):
                vds = open_virtual_dataset(
                    url,
                    reader_options={},
                    indexes={},
                )
        elif filetype == "hdf5":
            vds = open_virtual_dataset(
                url,
                group="science/LSAR/GCOV/grids/frequencyA",
                drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
                indexes={},
                reader_options={},
                backend=hdf_backend,
            )
            assert isinstance(vds, xr.Dataset)
        else:
            vds = open_virtual_dataset(url, indexes={})
            assert isinstance(vds, xr.Dataset)

    def test_virtualizarr_vs_local_nisar(self, hdf_backend):
        import fsspec

        # Open group directly from locally cached file with xarray
        url = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/GCOV/ALOS1_Rosamond_20081012/NISAR_L2_PR_GCOV_001_005_A_219_4020_SHNA_A_20081012T060910_20081012T060926_P01101_F_N_J_001.h5"
        tmpfile = fsspec.open_local(
            f"filecache::{url}", filecache=dict(cache_storage="/tmp", same_names=True)
        )
        hdf_group = "science/LSAR/GCOV/grids/frequencyA"
        dsXR = xr.open_dataset(
            tmpfile,
            engine="h5netcdf",
            group=hdf_group,
            drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
            phony_dims="access",
        )

        # save group reference file via virtualizarr, then open with engine="kerchunk"
        vds = open_virtual_dataset(
            tmpfile,
            group=hdf_group,
            indexes={},
            drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
            backend=hdf_backend,
        )
        tmpref = "/tmp/cmip6.json"
        vds.virtualize.to_kerchunk(tmpref, format="json")
        dsV = xr.open_dataset(tmpref, engine="kerchunk")

        # xrt.assert_identical(dsXR, dsV) #Attribute order changes
        xrt.assert_equal(dsXR, dsV)


@requires_kerchunk
class TestLoadVirtualDataset:
    @pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
    def test_loadable_variables(self, netcdf4_file, hdf_backend):
        vars_to_load = ["air", "time"]
        vds = open_virtual_dataset(
            netcdf4_file,
            loadable_variables=vars_to_load,
            indexes={},
            backend=hdf_backend,
        )

        for name in vds.variables:
            if name in vars_to_load:
                assert isinstance(vds[name].data, np.ndarray), name
            else:
                assert isinstance(vds[name].data, ManifestArray), name

        full_ds = xr.open_dataset(netcdf4_file, decode_times=True)

        for name in full_ds.variables:
            if name in vars_to_load:
                xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    def test_explicit_filetype(self, netcdf4_file):
        with pytest.raises(ValueError):
            open_virtual_dataset(netcdf4_file, filetype="unknown")

        with pytest.raises(NotImplementedError):
            open_virtual_dataset(netcdf4_file, filetype="grib")

    def test_explicit_filetype_and_backend(self, netcdf4_file):
        with pytest.raises(ValueError):
            open_virtual_dataset(
                netcdf4_file, filetype="hdf", backend=HDFVirtualBackend
            )

    @pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
    def test_group_kwarg(self, hdf5_groups_file, hdf_backend):
        if hdf_backend == HDFVirtualBackend:
            with pytest.raises(NotImplementedError, match="Nested groups"):
                open_virtual_dataset(hdf5_groups_file, backend=hdf_backend)
            with pytest.raises(KeyError, match="doesn't exist"):
                open_virtual_dataset(
                    hdf5_groups_file, group="doesnt_exist", backend=hdf_backend
                )
        if hdf_backend == HDF5VirtualBackend:
            with pytest.raises(ValueError, match="Multiple HDF Groups found"):
                open_virtual_dataset(hdf5_groups_file)
            with pytest.raises(ValueError, match="not found in"):
                open_virtual_dataset(hdf5_groups_file, group="doesnt_exist")

        vars_to_load = ["air", "time"]
        vds = open_virtual_dataset(
            hdf5_groups_file,
            group="test/group",
            loadable_variables=vars_to_load,
            indexes={},
            backend=hdf_backend,
        )
        full_ds = xr.open_dataset(
            hdf5_groups_file,
            group="test/group",
        )
        for name in full_ds.variables:
            if name in vars_to_load:
                xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    @pytest.mark.xfail(reason="patches a function which no longer exists")
    @patch("virtualizarr.translators.kerchunk.read_kerchunk_references_from_file")
    def test_open_virtual_dataset_passes_expected_args(
        self, mock_read_kerchunk, netcdf4_file
    ):
        reader_options = {"option1": "value1", "option2": "value2"}
        open_virtual_dataset(netcdf4_file, indexes={}, reader_options=reader_options)
        args = {
            "filepath": netcdf4_file,
            "filetype": None,
            "group": None,
            "reader_options": reader_options,
        }
        mock_read_kerchunk.assert_called_once_with(**args)

    @pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
    def test_open_dataset_with_empty(self, hdf5_empty, tmpdir, hdf_backend):
        vds = open_virtual_dataset(hdf5_empty, backend=hdf_backend)
        assert vds.empty.dims == ()
        assert vds.empty.attrs == {"empty": "true"}

    @pytest.mark.parametrize("hdf_backend", [HDF5VirtualBackend, HDFVirtualBackend])
    def test_open_dataset_with_scalar(self, hdf5_scalar, tmpdir, hdf_backend):
        vds = open_virtual_dataset(hdf5_scalar, backend=hdf_backend)
        assert vds.scalar.dims == ()
        assert vds.scalar.attrs == {"scalar": "true"}


@requires_kerchunk
@pytest.mark.parametrize(
    "reference_format",
    ["json", "parquet", "invalid"],
)
def test_open_virtual_dataset_existing_kerchunk_refs(
    tmp_path, netcdf4_virtual_dataset, reference_format
):
    example_reference_dict = netcdf4_virtual_dataset.virtualize.to_kerchunk(
        format="dict"
    )

    if reference_format == "invalid":
        # Test invalid file format leads to ValueError
        ref_filepath = tmp_path / "ref.csv"
        with open(ref_filepath.as_posix(), mode="w") as of:
            of.write("tmp")

        with pytest.raises(ValueError):
            open_virtual_dataset(
                filepath=ref_filepath.as_posix(), filetype="kerchunk", indexes={}
            )

    else:
        # Test valid json and parquet reference formats

        if reference_format == "json":
            ref_filepath = tmp_path / "ref.json"

            import ujson

            with open(ref_filepath, "w") as json_file:
                ujson.dump(example_reference_dict, json_file)

        if reference_format == "parquet":
            from kerchunk.df import refs_to_dataframe

            ref_filepath = tmp_path / "ref.parquet"
            refs_to_dataframe(fo=example_reference_dict, url=ref_filepath.as_posix())

        vds = open_virtual_dataset(
            filepath=ref_filepath.as_posix(), filetype="kerchunk", indexes={}
        )

        # Inconsistent results! https://github.com/TomNicholas/VirtualiZarr/pull/73#issuecomment-2040931202
        # assert vds.virtualize.to_kerchunk(format='dict') == example_reference_dict
        refs = vds.virtualize.to_kerchunk(format="dict")
        expected_refs = netcdf4_virtual_dataset.virtualize.to_kerchunk(format="dict")
        assert refs["refs"]["air/0.0.0"] == expected_refs["refs"]["air/0.0.0"]
        assert refs["refs"]["lon/0"] == expected_refs["refs"]["lon/0"]
        assert refs["refs"]["lat/0"] == expected_refs["refs"]["lat/0"]
        assert refs["refs"]["time/0"] == expected_refs["refs"]["time/0"]

        assert list(vds) == list(netcdf4_virtual_dataset)
        assert set(vds.coords) == set(netcdf4_virtual_dataset.coords)
        assert set(vds.variables) == set(netcdf4_virtual_dataset.variables)


@requires_kerchunk
def test_notimplemented_read_inline_refs(tmp_path, netcdf4_inlined_ref):
    # For now, we raise a NotImplementedError if we read existing references that have inlined data
    # https://github.com/zarr-developers/VirtualiZarr/pull/251#pullrequestreview-2361916932

    ref_filepath = tmp_path / "ref.json"

    import ujson

    with open(ref_filepath, "w") as json_file:
        ujson.dump(netcdf4_inlined_ref, json_file)

    with pytest.raises(NotImplementedError):
        open_virtual_dataset(
            filepath=ref_filepath.as_posix(), filetype="kerchunk", indexes={}
        )
