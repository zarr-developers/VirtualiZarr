import xarray as xr

from virtualizarr import open_virtual_dataarray
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import requires_asynctiff


@requires_asynctiff
def test_read_single_band_tiff():
    filepath = "tests/data/daymet_v4_swe_annavg_hi_2023.tif"
    vda = open_virtual_dataarray(filepath)

    assert isinstance(vda, xr.DataArray)
    assert isinstance(vda.data, ManifestArray)
