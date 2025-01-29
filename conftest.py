from pathlib import Path
from typing import Any, Callable, Mapping, Optional

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
def empty_netcdf4_file(tmp_path: Path) -> str:
    filepath = tmp_path / "empty.nc"

    # Set up example xarray dataset
    with xr.Dataset() as ds:  # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_file(tmp_path: Path) -> str:
    filepath = tmp_path / "air.nc"

    # Set up example xarray dataset
    with xr.tutorial.open_dataset("air_temperature") as ds:
        # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_file_with_data_in_multiple_groups(tmp_path: Path) -> str:
    filepath = tmp_path / "test.nc"

    ds1 = xr.DataArray([1, 2, 3], name="foo").to_dataset()
    ds1.to_netcdf(filepath)
    ds2 = xr.DataArray([4, 5], name="bar").to_dataset()
    ds2.to_netcdf(filepath, group="subgroup", mode="a")

    return str(filepath)


@pytest.fixture
def netcdf4_files_factory(tmp_path: Path) -> Callable:
    def create_netcdf4_files(
        encoding: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[str, str]:
        filepath1 = tmp_path / "air1.nc"
        filepath2 = tmp_path / "air2.nc"

        with xr.tutorial.open_dataset("air_temperature") as ds:
            # Split dataset into two parts
            ds1 = ds.isel(time=slice(None, 1460))
            ds2 = ds.isel(time=slice(1460, None))

            # Save datasets to disk as NetCDF in the temporary directory with the provided encoding
            ds1.to_netcdf(filepath1, encoding=encoding)
            ds2.to_netcdf(filepath2, encoding=encoding)

        return str(filepath1), str(filepath2)

    return create_netcdf4_files


@pytest.fixture
def netcdf4_file_with_2d_coords(tmp_path: Path) -> str:
    filepath = tmp_path / "ROMS_example.nc"

    with xr.tutorial.open_dataset("ROMS_example") as ds:
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_virtual_dataset(netcdf4_file):
    from virtualizarr import open_virtual_dataset

    return open_virtual_dataset(netcdf4_file, indexes={})


@pytest.fixture
def netcdf4_inlined_ref(netcdf4_file):
    from kerchunk.hdf import SingleHdf5ToZarr

    return SingleHdf5ToZarr(netcdf4_file, inline_threshold=1000).translate()


@pytest.fixture
def hdf5_groups_file(tmp_path: Path) -> str:
    filepath = tmp_path / "air.nc"

    # Set up example xarray dataset
    with xr.tutorial.open_dataset("air_temperature") as ds:
        # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4", group="test/group")

    return str(filepath)


@pytest.fixture
def hdf5_empty(tmp_path: Path) -> str:
    filepath = tmp_path / "empty.nc"

    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("empty", shape=(), dtype="float32")
        dataset.attrs["empty"] = "true"

    return str(filepath)


@pytest.fixture
def hdf5_scalar(tmp_path: Path) -> str:
    filepath = tmp_path / "scalar.nc"

    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("scalar", data=0.1, dtype="float32")
        dataset.attrs["scalar"] = "true"

    return str(filepath)


@pytest.fixture
def simple_netcdf4(tmp_path: Path) -> str:
    filepath = tmp_path / "simple.nc"

    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4)
    var = Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    ds.to_netcdf(filepath)

    return str(filepath)
