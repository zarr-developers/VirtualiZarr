from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

from virtualizarr import open_virtual_dataarray
from virtualizarr.manifests import ManifestArray
from virtualizarr.readers import TIFFVirtualBackend
from virtualizarr.tests import requires_asynctiff


@requires_asynctiff
def test_read_single_band_tiff():
    from async_tiff.store import LocalStore as AsyncTiffLocalStore
    from obstore.store import LocalStore

    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    filepath = str(
        repo_root / "data/LC08_L2SP_046027_20201229_20210310_02_T2_SR_B2.TIF"
    )
    store = AsyncTiffLocalStore()
    backend_kwargs = {"store": store, "ifd": 0}
    vda = open_virtual_dataarray(
        filepath, backend=TIFFVirtualBackend, virtual_backend_kwargs=backend_kwargs
    )
    assert isinstance(vda, xr.DataArray)
    assert isinstance(vda.data, ManifestArray)
    store = LocalStore()
    stores = {"file://": store}
    da_actual = vda.virtualize.to_xarray(stores).load()
    da_expected = rioxarray.open_rasterio(filepath)
    np.testing.assert_allclose(da_actual.data, da_expected.data.squeeze())
