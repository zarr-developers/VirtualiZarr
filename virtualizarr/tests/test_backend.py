import functools
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray import Dataset, open_dataset
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.backends.hdf import HDFBackend
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import (
    has_astropy,
    parametrize_over_hdf_backends,
    requires_dask,
    requires_hdf5plugin,
    requires_imagecodecs,
    requires_lithops,
    requires_network,
)
from virtualizarr.tests.utils import obstore_http, obstore_local, obstore_s3

# @requires_scipy
# def test_automatically_determine_filetype_netcdf3_netcdf4():
    # # test the NetCDF3 vs NetCDF4 automatic file type selection

    # ds = xr.Dataset({"a": (["x"], [0, 1])})
    # netcdf3_file_path = "/tmp/netcdf3.nc"
    # netcdf4_file_path = "/tmp/netcdf4.nc"

    # # write two version of NetCDF
    # ds.to_netcdf(netcdf3_file_path, engine="scipy", format="NETCDF3_CLASSIC")
    # ds.to_netcdf(netcdf4_file_path, engine="h5netcdf")

    # assert FileType("netcdf3") == automatically_determine_filetype(
        # filepath=netcdf3_file_path
    # )
    # assert FileType("hdf5") == automatically_determine_filetype(
        # filepath=netcdf4_file_path
    # )


# @pytest.mark.parametrize(
    # "filetype,headerbytes",
    # [
        # ("netcdf3", b"CDF"),
        # ("hdf5", b"\x89HDF"),
        # ("grib", b"GRIB"),
        # ("tiff", b"II*"),
        # ("fits", b"SIMPLE"),
    # ],
# )
# def test_valid_filetype_bytes(tmp_path, filetype, headerbytes):
    # filepath = tmp_path / "file.abc"
    # with open(filepath, "wb") as f:
        # f.write(headerbytes)
    # assert FileType(filetype) == automatically_determine_filetype(filepath=filepath)


# def test_notimplemented_filetype(tmp_path):
    # for headerbytes in [b"JUNK", b"\x0e\x03\x13\x01"]:
        # filepath = tmp_path / "file.abc"
        # with open(filepath, "wb") as f:
            # f.write(headerbytes)
        # with pytest.raises(NotImplementedError):
            # automatically_determine_filetype(filepath=filepath)


# def test_FileType():
    # # tests if FileType converts user supplied strings to correct filetype
    # assert "netcdf3" == FileType("netcdf3").name
    # assert "netcdf4" == FileType("netcdf4").name
    # assert "hdf4" == FileType("hdf4").name
    # assert "hdf5" == FileType("hdf5").name
    # assert "grib" == FileType("grib").name
    # assert "tiff" == FileType("tiff").name
    # assert "fits" == FileType("fits").name
    # with pytest.raises(ValueError):
        # FileType(None)


class TestOpenVirtualDatasetIndexes:
    @pytest.mark.xfail(reason="not yet implemented")
    def test_specify_no_indexes(self, netcdf4_file):
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        vds = open_virtual_dataset(
            filepath=netcdf4_file,
            object_reader=object_reader,
            backend=hdf_backend, 
            indexes={}
        )
        assert vds.indexes == {}

    @requires_hdf5plugin
    @requires_imagecodecs
    def test_create_default_indexes_for_loadable_variables(
        self, netcdf4_file
    ):
        loadable_variables = ["time", "lat"]
        
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        with (
            open_virtual_dataset(
                filepath=netcdf4_file,
                object_reader=object_reader,
                backend=hdf_backend,
                indexes=None,
                loadable_variables=loadable_variables,
            ) as vds,
            open_dataset(netcdf4_file, decode_times=True) as ds,
        ):
            # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
            assert index_mappings_equal(vds.xindexes, ds[loadable_variables].xindexes)


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


@requires_hdf5plugin
@requires_imagecodecs
def test_cftime_index(tmp_path: Path):
    """Ensure a virtual dataset contains the same indexes as an Xarray dataset"""
    # Note: Test was created to debug: https://github.com/zarr-developers/VirtualiZarr/issues/168
    filepath = str(tmp_path / "tmp.nc")
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
    ds.to_netcdf(filepath)

    object_reader = obstore_local(filepath=filepath)
    hdf_backend = HDFBackend()
    with open_virtual_dataset(
        filepath=filepath,
        object_reader=object_reader,
        backend=hdf_backend,
        loadable_variables=["time", "lat", "lon"],
    ) as vds:
        # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
        assert index_mappings_equal(vds.xindexes, ds.xindexes)
        assert list(ds.coords) == list(vds.coords)
        assert vds.dims == ds.dims
        assert vds.attrs == ds.attrs


class TestOpenVirtualDatasetAttrs:
    def test_drop_array_dimensions(self, netcdf4_file):
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        # regression test for GH issue #150
        vds = open_virtual_dataset(
            filepath=netcdf4_file, 
            object_reader=object_reader,
            backend=hdf_backend,
        )
        assert "_ARRAY_DIMENSIONS" not in vds["air"].attrs

    def test_coordinate_variable_attrs_preserved(self, netcdf4_file):
        # regression test for GH issue #155
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        vds = open_virtual_dataset(
            filepath=netcdf4_file, 
            object_reader=object_reader,
            backend=hdf_backend,
        )
        assert vds["lat"].attrs == {
            "standard_name": "latitude",
            "long_name": "Latitude",
            "units": "degrees_north",
            "axis": "Y",
        }


class TestDetermineCoords:
    def test_infer_one_dimensional_coords(self, netcdf4_file):
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=netcdf4_file, 
            object_reader=object_reader,
            backend=hdf_backend,
        ) as vds:
            assert set(vds.coords) == {"time", "lat", "lon"}

    def test_var_attr_coords(self, netcdf4_file_with_2d_coords):
        object_reader = obstore_local(filepath=netcdf4_file_with_2d_coords)
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=netcdf4_file_with_2d_coords, 
            object_reader=object_reader,
            backend=hdf_backend,

        ) as vds:
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


@requires_network
class TestReadFromS3:
    @pytest.mark.parametrize(
        "indexes",
        [
            None,
            pytest.param({}, marks=pytest.mark.xfail(reason="not implemented")),
        ],
        ids=["None index", "empty dict index"],
    )
    def test_anon_read_s3(self, indexes):
        """Parameterized tests for empty vs supplied indexes and filetypes."""
        # TODO: Switch away from this s3 url after minIO is implemented.
        filepath = "s3://carbonplan-share/virtualizarr/local.nc"
        object_reader = obstore_s3(filepath=filepath, region="us-west-2")
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=filepath,
            object_reader=object_reader,
            indexes=indexes,
            backend=hdf_backend,
        ) as vds:
            assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}

            assert isinstance(vds["air"].data, ManifestArray)
            for name in ["time", "lat", "lon"]:
                assert isinstance(vds[name].data, np.ndarray)


# @requires_network
class TestReadFromURL:
    @pytest.mark.parametrize(
        "filetype, url",
        [
            (
                "grib",
                "https://github.com/pydata/xarray-data/raw/master/era5-2mt-2019-03-uk.grib",
            ),
            pytest.param(
                "netcdf3",
                "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
                marks=pytest.mark.xfail(
                    reason="Big endian not yet supported by zarr-python 3.0"
                ),  # https://github.com/zarr-developers/zarr-python/issues/2324
            ),
            (
                "netcdf4",
                "https://github.com/pydata/xarray-data/raw/master/ROMS_example.nc",
            ),
            pytest.param(
                "hdf4",
                "https://github.com/corteva/rioxarray/raw/master/test/test_data/input/MOD09GA.A2008296.h14v17.006.2015181011753.hdf",
                marks=pytest.mark.skip(reason="often times out"),
            ),
            pytest.param(
                "hdf5",
                "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/GCOV/ALOS1_Rosamond_20081012/NISAR_L2_PR_GCOV_001_005_A_219_4020_SHNA_A_20081012T060910_20081012T060926_P01101_F_N_J_001.h5",
                marks=pytest.mark.skip(reason="often times out"),
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
                marks=[
                    pytest.mark.skipif(
                        not has_astropy, reason="package astropy is not available"
                    ),
                    pytest.mark.xfail(
                        reason="Big endian not yet supported by zarr-python 3.0"
                    ),  # https://github.com/zarr-developers/zarr-python/issues/2324
                ],
            ),
            (
                "jpg",
                "https://github.com/rasterio/rasterio/raw/main/tests/data/389225main_sw_1965_1024.jpg",
            ),
        ],
    )
    def test_read_from_url(self, filetype, url):
        if filetype == "netcdf3":
            pytest.importorskip("scipy")
        # if filetype in ["grib", "jpg", "hdf4"]:
            # with pytest.raises(NotImplementedError):
                # open_virtual_dataset(url, reader_options={})
        elif filetype == "hdf5":
            object_reader = obstore_http(filepath=url)
            hdf_backend = HDFBackend(
                group="science/LSAR/GCOV/grids/frequencyA",
                drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
            )
            with open_virtual_dataset(
                filepath=url,
                object_reader=object_reader,
                backend=hdf_backend,
            ) as vds:
                assert isinstance(vds, xr.Dataset)
        # else:
            # backend_args = obstore_http(filepath=url)
            # with open_virtual_dataset(url) as vds:
                # assert isinstance(vds, xr.Dataset)

    @pytest.mark.skip(reason="often times out, as nisar file is 200MB")
    def test_virtualizarr_vs_local_nisar(self, hdf_backend):
        import fsspec

        # Open group directly from locally cached file with xarray
        url = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/GCOV/ALOS1_Rosamond_20081012/NISAR_L2_PR_GCOV_001_005_A_219_4020_SHNA_A_20081012T060910_20081012T060926_P01101_F_N_J_001.h5"
        tmpfile = fsspec.open_local(
            f"filecache::{url}", filecache=dict(cache_storage="/tmp", same_names=True)
        )
        assert isinstance(tmpfile, str)  # make type-checkers happy
        hdf_group = "science/LSAR/GCOV/grids/frequencyA"

        with (
            xr.open_dataset(
                tmpfile,
                engine="h5netcdf",
                group=hdf_group,
                drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
                phony_dims="access",
            ) as dsXR,
            # save group reference file via virtualizarr, then open with engine="kerchunk"
            open_virtual_dataset(
                tmpfile,
                group=hdf_group,
                drop_variables=["listOfCovarianceTerms", "listOfPolarizations"],
                backend=hdf_backend,
            ) as vds,
        ):
            tmpref = "/tmp/cmip6.json"
            vds.virtualize.to_kerchunk(tmpref, format="json")

            with xr.open_dataset(tmpref, engine="kerchunk") as dsV:
                # xrt.assert_identical(dsXR, dsV) #Attribute order changes
                xrt.assert_equal(dsXR, dsV)


class TestOpenVirtualDatasetHDFGroup:
    def test_open_empty_group(self, empty_netcdf4_file):
        object_reader = obstore_local(filepath=empty_netcdf4_file)
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=empty_netcdf4_file, 
            object_reader=object_reader,
            backend=hdf_backend,
        ) as vds:
            assert isinstance(vds, xr.Dataset)
            expected = Dataset()
            xrt.assert_identical(vds, expected)

    def test_open_subgroup(self, netcdf4_file_with_data_in_multiple_groups):
        object_reader = obstore_local(filepath=netcdf4_file_with_data_in_multiple_groups)
        hdf_backend = HDFBackend(
            group = "subgroup"
        )
        with open_virtual_dataset(
            filepath=netcdf4_file_with_data_in_multiple_groups,
            object_reader=object_reader,
            backend=hdf_backend,
        ) as vds:
            assert list(vds.variables) == ["bar"]
            assert isinstance(vds["bar"].data, ManifestArray)
            assert vds["bar"].shape == (2,)

    @pytest.mark.parametrize("group", ["", None])
    def test_open_root_group(
        self,
        netcdf4_file_with_data_in_multiple_groups,
        group,
    ):
        object_reader = obstore_local(filepath=netcdf4_file_with_data_in_multiple_groups)
        hdf_backend = HDFBackend(group=group)
        with open_virtual_dataset(
            filepath=netcdf4_file_with_data_in_multiple_groups,
            object_reader=object_reader,
            backend=hdf_backend,
        ) as vds:
            assert list(vds.variables) == ["foo"]
            assert isinstance(vds["foo"].data, ManifestArray)
            assert vds["foo"].shape == (3,)


@requires_hdf5plugin
@requires_imagecodecs
class TestLoadVirtualDataset:
    @pytest.mark.parametrize(
        "loadable_variables, expected_loadable_variables",
        [
            ([], []),
            (["time"], ["time"]),
            (["air", "time"], ["air", "time"]),
            (None, ["lat", "lon", "time"]),
        ],
    )
    def test_loadable_variables(
        self, netcdf4_file, loadable_variables, expected_loadable_variables
    ):
        object_reader = obstore_local(filepath=netcdf4_file)
        hdf_backend = HDFBackend()
        with (
            open_virtual_dataset(
                filepath=netcdf4_file,
                object_reader=object_reader,
                loadable_variables=loadable_variables,
                backend=hdf_backend,
            ) as vds,
            xr.open_dataset(netcdf4_file, decode_times=True) as ds,
        ):
            assert set(vds.variables) == set(ds.variables)
            assert set(vds.coords) == set(ds.coords)

            virtual_variables = {
                name: var
                for name, var in vds.variables.items()
                if isinstance(var.data, ManifestArray)
            }
            actual_loadable_variables = {
                name: var
                for name, var in vds.variables.items()
                if not isinstance(var.data, ManifestArray)
            }

            assert set(actual_loadable_variables) == set(expected_loadable_variables)

            for var in virtual_variables.values():
                assert isinstance(var.data, ManifestArray)

            for name, var in ds.variables.items():
                if name in actual_loadable_variables:
                    xrt.assert_identical(vds.variables[name], ds.variables[name])

    # def test_explicit_filetype(self, netcdf4_file):
        # with pytest.raises(ValueError):
            # open_virtual_dataset(netcdf4_file, filetype="unknown")

        # with pytest.raises(ValueError):
            # open_virtual_dataset(netcdf4_file, filetype=ManifestArray)  # type: ignore

        # with pytest.raises(NotImplementedError):
            # open_virtual_dataset(netcdf4_file, filetype="grib")

        # with open_virtual_dataset(netcdf4_file, filetype="netCDF4"):
            # pass

    # def test_explicit_filetype_and_backend(self, netcdf4_file):
        # with pytest.raises(ValueError):
            # open_virtual_dataset(
                # netcdf4_file, filetype="hdf", backend=HDFVirtualBackend
            # )

    def test_group_kwarg(self, hdf5_groups_file):
        object_reader = obstore_local(filepath=hdf5_groups_file)
        hdf_backend = HDFBackend(group="doesnt_exist")
        with pytest.raises(KeyError, match="doesn't exist"):
            with open_virtual_dataset(
                filepath=hdf5_groups_file,
                object_reader=object_reader,
                backend=hdf_backend,
            ):
                pass

        hdf_backend = HDFBackend(group="test/group")
        vars_to_load = ["air", "time"]
        with (
            open_virtual_dataset(
                filepath=hdf5_groups_file,
                object_reader=object_reader,
                loadable_variables=vars_to_load,
                backend=hdf_backend,
            ) as vds,
                xr.open_dataset(hdf5_groups_file, group="test/group") as full_ds,
        ):
            for name in full_ds.variables:
                if name in vars_to_load:
                    xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    # @pytest.mark.xfail(reason="patches a function which no longer exists")
    # @patch("virtualizarr.translators.kerchunk.read_kerchunk_references_from_file")
    # def test_open_virtual_dataset_passes_expected_args(
        # self, mock_read_kerchunk, netcdf4_file
    # ):
        # reader_options = {"option1": "value1", "option2": "value2"}
        # with open_virtual_dataset(netcdf4_file, reader_options=reader_options):
            # pass
        # args = {
            # "filepath": netcdf4_file,
            # "filetype": None,
            # "group": None,
            # "reader_options": reader_options,
        # }
        # mock_read_kerchunk.assert_called_once_with(**args)

    def test_open_dataset_with_empty(self, hdf5_empty):
        object_reader = obstore_local(filepath=hdf5_empty)
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=hdf5_empty,
            object_reader=object_reader,
            backend=hdf_backend
        ) as vds:
            assert vds.empty.dims == ()
            assert vds.empty.attrs == {"empty": "true"}

    def test_open_dataset_with_scalar(self, hdf5_scalar):
        object_reader = obstore_local(filepath=hdf5_scalar)
        hdf_backend = HDFBackend()
        with open_virtual_dataset(
            filepath=hdf5_scalar,
            object_reader=object_reader,
            backend=hdf_backend
        ) as vds:
            assert vds.scalar.dims == ()
            assert vds.scalar.attrs == {"scalar": "true"}


preprocess_func = functools.partial(
    xr.Dataset.rename_vars,
    air="nair",
)


@requires_hdf5plugin
@requires_imagecodecs
@parametrize_over_hdf_backends
class TestOpenVirtualMFDataset:
    @pytest.mark.parametrize("invalid_parallel_kwarg", ["ray", Dataset])
    def test_invalid_parallel_kwarg(
        self, netcdf4_files_factory, invalid_parallel_kwarg, hdf_backend
    ):
        filepath1, filepath2 = netcdf4_files_factory()

        with pytest.raises(ValueError, match="Unrecognized argument"):
            open_virtual_mfdataset(
                [filepath1, filepath2],
                combine="nested",
                concat_dim="time",
                backend=hdf_backend,
                parallel=invalid_parallel_kwarg,
            )

    @pytest.mark.parametrize(
        "parallel",
        [
            False,
            ThreadPoolExecutor,
            pytest.param("dask", marks=requires_dask),
            pytest.param("lithops", marks=requires_lithops),
        ],
    )
    @pytest.mark.parametrize(
        "preprocess",
        [
            None,
            preprocess_func,
        ],
    )
    def test_parallel_open(
        self, netcdf4_files_factory, hdf_backend, parallel, preprocess
    ):
        filepath1, filepath2 = netcdf4_files_factory()
        vds1 = open_virtual_dataset(filepath1, backend=hdf_backend)
        vds2 = open_virtual_dataset(filepath2, backend=hdf_backend)

        expected_vds = xr.concat([vds1, vds2], dim="time")
        if preprocess:
            expected_vds = preprocess_func(expected_vds)

        # test combine nested, which doesn't use in-memory indexes
        combined_vds = open_virtual_mfdataset(
            [filepath1, filepath2],
            combine="nested",
            concat_dim="time",
            backend=hdf_backend,
            parallel=parallel,
            preprocess=preprocess,
        )
        xrt.assert_identical(combined_vds, expected_vds)

        # test combine by coords using in-memory indexes
        combined_vds = open_virtual_mfdataset(
            [filepath1, filepath2],
            combine="by_coords",
            backend=hdf_backend,
            parallel=parallel,
            preprocess=preprocess,
        )
        xrt.assert_identical(combined_vds, expected_vds)

        # test combine by coords again using in-memory indexes but for a glob
        file_glob = Path(filepath1).parent.glob("air*.nc")
        combined_vds = open_virtual_mfdataset(
            file_glob,
            combine="by_coords",
            backend=hdf_backend,
            parallel=parallel,
            preprocess=preprocess,
        )
        xrt.assert_identical(combined_vds, expected_vds)
