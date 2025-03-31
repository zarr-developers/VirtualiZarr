import numpy as np
import xarray as xr

from virtualizarr.readers import TIFFVirtualBackend
from virtualizarr.tests import requires_asynctiff, requires_rioxarray


@requires_asynctiff
@requires_rioxarray
def test_read_geotiff(geotiff_file):
    import rioxarray
    from obstore.store import LocalStore

    store = LocalStore()
    kwargs = {
        "file_id": "file://",
        "store": store,
    }
    ds = TIFFVirtualBackend.open_virtual_dataset(
        filepath=geotiff_file, group="0", virtual_backend_kwargs=kwargs
    )
    assert isinstance(ds, xr.Dataset)
    da_expected = rioxarray.open_rasterio(geotiff_file)
    np.testing.assert_allclose(ds["0"].data, da_expected.data.squeeze())
