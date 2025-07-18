import numpy as np
import xarray as xr
from virtual_tiff import TIFFParser

from virtualizarr.tests import requires_asynctiff, requires_rioxarray


@requires_asynctiff
@requires_rioxarray
def test_read_geotiff(geotiff_file):
    import rioxarray
    from obstore.store import LocalStore

    parser = TIFFParser(ifd=0)
    ms = parser(file_url=f"file://{geotiff_file}", object_store=LocalStore())
    ds = xr.open_dataset(ms, engine="zarr", consolidated=False, zarr_format=3).load()
    assert isinstance(ds, xr.Dataset)
    expected = rioxarray.open_rasterio(geotiff_file).data.squeeze()
    observed = ds["0"].data.squeeze()
    np.testing.assert_allclose(observed, expected)
