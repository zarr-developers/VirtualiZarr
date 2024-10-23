import h5py
import numpy as np
import pytest
import xarray as xr
from xarray.core.variable import Variable


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def pytest_runtest_setup(item):
    # based on https://stackoverflow.com/questions/47559524
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )


@pytest.fixture
def netcdf4_file(tmpdir):
    # Set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature")

    # Save it to disk as netCDF (in temporary directory)
    filepath = f"{tmpdir}/air.nc"
    ds.to_netcdf(filepath, format="NETCDF4")
    ds.close()

    return filepath


@pytest.fixture
def netcdf4_virtual_dataset(netcdf4_file):
    from virtualizarr import open_virtual_dataset

    return open_virtual_dataset(netcdf4_file, indexes={})


@pytest.fixture
def netcdf4_inlined_ref(netcdf4_file):
    from kerchunk.hdf import SingleHdf5ToZarr

    return SingleHdf5ToZarr(netcdf4_file, inline_threshold=1000).translate()


@pytest.fixture
def hdf5_groups_file(tmpdir):
    # Set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature")

    # Save it to disk as netCDF (in temporary directory)
    filepath = f"{tmpdir}/air.nc"
    ds.to_netcdf(filepath, format="NETCDF4", group="test/group")
    ds.close()

    return filepath


@pytest.fixture
def netcdf4_files(tmpdir):
    # Set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature")

    # split inrto equal chunks so we can concatenate them back together later
    ds1 = ds.isel(time=slice(None, 1460))
    ds2 = ds.isel(time=slice(1460, None))

    # Save it to disk as netCDF (in temporary directory)
    filepath1 = f"{tmpdir}/air1.nc"
    filepath2 = f"{tmpdir}/air2.nc"
    ds1.to_netcdf(filepath1)
    ds2.to_netcdf(filepath2)
    ds1.close()
    ds2.close()

    return filepath1, filepath2


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


@pytest.fixture
def simple_netcdf4(tmpdir):
    filepath = f"{tmpdir}/simple.nc"

    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4)
    var = Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})

    ds.to_netcdf(filepath)

    return filepath
