import pytest
import xarray as xr


def _xarray_subset():
    ds = xr.tutorial.open_dataset("air_temperature", chunks={})
    return ds.isel(time=slice(0, 10), lat=slice(0, 9), lon=slice(0, 18)).chunk(
        {"time": 5}
    )


@pytest.fixture(params=[2, 3])
def zarr_store(tmpdir, request):
    ds = _xarray_subset()
    filepath = f"{tmpdir}/air.zarr"
    ds.to_zarr(filepath, zarr_format=request.param)
    ds.close()
    return filepath
