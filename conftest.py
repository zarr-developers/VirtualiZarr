import h5py
import pytest
import xarray as xr


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def generate_test_filepath_dict() -> dict:
    return {
        "full_time_netcdf4": "virtualizarr/tests/data/test_ds_netcdf4_full_time.nc",
        "time_1_netcdf4": "virtualizarr/tests/data/test_ds_netcdf4_split_time1.nc",
        "time_2_netcdf4": "virtualizarr/tests/data/test_ds_netcdf4_split_time2.nc",
        "netcdf3": "virtualizarr/tests/data/test_ds_netcdf3.nc",
        "netcdf4_group": "virtualizarr/tests/data/test_ds_netcdf4_group.nc",
        "netcdf4_non_standard_time": "virtualizarr/tests/data/test_ds_non_datetime_time.nc",
    }


def generate_small_xr_datasets():
    """This function can be used to re-generate the locally stored dataset for testing
    It's 43kB instead of the full 31MB air_temperature tutorial dataset"""
    import numpy as np

    # building our test dataset from the air_temp dataset, but saving a subset
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(0, 4))

    lats = np.arange(-90, 90, 1)
    lons = np.arange(-180, 180, 1)

    data = np.random.randint(0, 2, size=(4, 180, 360), dtype=np.int16)

    # create a dataset with non-standard time
    non_standard_date_ds = xr.Dataset(
        data_vars=dict(air=(["time", "lat", "lon"], data)),
        coords=dict(time=[0, 1, 2, 3], lat=lats, lon=lons),
    )

    # Add attributes to the time coordinate
    non_standard_date_ds.time.attrs["units"] = "days since '2000-01-01'"
    non_standard_date_ds.time.attrs["calendar"] = "standard"

    # write datasets
    ds.to_netcdf("virtualizarr/tests/data/test_ds_netcdf4_full_time.nc")
    ds.isel(time=slice(0, 2)).to_netcdf(
        "virtualizarr/tests/data/test_ds_netcdf4_split_time1.nc"
    )
    ds.isel(time=slice(2, 4)).to_netcdf(
        "virtualizarr/tests/data/test_ds_netcdf4_split_time2.nc"
    )

    ds.to_netcdf("virtualizarr/tests/data/test_ds_netcdf3.nc", engine="scipy")
    ds.to_netcdf("virtualizarr/tests/data/test_ds_netcdf4_group.nc", group="test/group")

    non_standard_date_ds.to_netcdf(
        "virtualizarr/tests/data/test_ds_non_datetime_time.nc"
    )


def pytest_runtest_setup(item):
    # based on https://stackoverflow.com/questions/47559524
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )


@pytest.fixture
def netcdf4_file() -> str:
    return generate_test_filepath_dict()["full_time_netcdf4"]


@pytest.fixture
def hdf5_groups_file() -> str:
    return generate_test_filepath_dict()["netcdf4_group"]


@pytest.fixture
def netcdf4_files():
    test_filepath_dict = generate_test_filepath_dict()
    return test_filepath_dict["time_1_netcdf4"], test_filepath_dict["time_2_netcdf4"]


@pytest.fixture
def hdf5_empty(tmpdir):
    filepath = f"{tmpdir}/empty.nc"
    f = h5py.File(filepath, "w")
    dataset = f.create_dataset("empty", shape=(), dtype="float32")
    dataset.attrs["empty"] = "true"
    return filepath


@pytest.fixture
def hdf5_scalar(tmpdir):
    filepath = f"{tmpdir}/scalar.nc"
    f = h5py.File(filepath, "w")
    dataset = f.create_dataset("scalar", data=0.1, dtype="float32")
    dataset.attrs["scalar"] = "true"
    return filepath
