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
def example_reference_dict() -> dict:
    return {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": '{"coordinates":"lat time lon"}',
            "air/0.0.0": ["tmp.nc", 9123, 10600],
            "air/.zarray": '{"shape":[4,25,53],"chunks":[4,25,53],"dtype":"<i2","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
            "air/.zattrs": '{"scale_factor":0.01,"_ARRAY_DIMENSIONS":["time","lat","lon"]}',
            "lat/0": ["tmp.nc", 4927, 100],
            "lat/.zarray": '{"shape":[25],"chunks":[25],"dtype":"<f4","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
            "lat/.zattrs": '{"axis":"Y","long_name":"Latitude","standard_name":"latitude","units":"degrees_north","_ARRAY_DIMENSIONS":["lat"]}',
            "time/0": ["tmp.nc", 23396, 16],
            "time/.zarray": '{"shape":[4],"chunks":[4],"dtype":"<f4","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
            "time/.zattrs": '{"calendar":"standard","long_name":"Time","standard_name":"time","units":"hours since 1800-01-01","_ARRAY_DIMENSIONS":["time"]}',
            "lon/0": ["tmp.nc", 23184, 212],
            "lon/.zarray": '{"shape":[53],"chunks":[53],"dtype":"<f4","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
            "lon/.zattrs": '{"axis":"X","long_name":"Longitude","standard_name":"longitude","units":"degrees_east","_ARRAY_DIMENSIONS":["lon"]}',
        },
    }
