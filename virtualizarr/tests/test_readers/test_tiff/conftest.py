from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def geotiff_file(tmp_path: Path) -> str:
    """Create a NetCDF4 file with air temperature data."""
    filepath = tmp_path / "air.tif"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.isel(time=0).rio.to_raster(filepath)
    return str(filepath)
