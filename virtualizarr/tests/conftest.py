import pytest
import xarray as xr


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky", action="store_true", help="runs flaky tests")
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
