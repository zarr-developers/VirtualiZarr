import pytest
import xarray as xr

@pytest.fixture
def netcdf4_file(tmpdir):
    # Set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature")

    # Save it to disk as netCDF (in temporary directory)
    filepath = f"{tmpdir}/air.nc"
    ds.to_netcdf(filepath)

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

    return filepath1, filepath2

@pytest.fixture
def concated_virtual_dataset_with_indexes(netcdf4_files):
        """Fixture to supply concatenated virtual dataset including indexes"""
        from virtualizarr import open_virtual_dataset

        filepath1, filepath2 = netcdf4_files

        vds1 = open_virtual_dataset(filepath1)
        vds2 = open_virtual_dataset(filepath2)

        return xr.combine_by_coords(
            [vds2, vds1],
        )
