from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

from virtualizarr.readers import TIFFVirtualBackend
from virtualizarr.tests import requires_asynctiff


@requires_asynctiff
def test_read_single_band_tiff():
    from obstore.store import LocalStore

    current_file_path = Path(__file__).resolve()
    repo_root = current_file_path.parent.parent
    filepath = str(
        repo_root / "data/LC08_L2SP_046027_20201229_20210310_02_T2_SR_B2.TIF"
    )
    store = LocalStore()
    kwargs = {
        "file_id": "file://",
        "store": store,
    }
    ds = TIFFVirtualBackend.open_virtual_dataset(
        filepath=filepath, group="0", virtual_backend_kwargs=kwargs
    )
    assert isinstance(ds, xr.Dataset)
    da_expected = rioxarray.open_rasterio(filepath)
    np.testing.assert_allclose(ds["0"].data, da_expected.data.squeeze())
