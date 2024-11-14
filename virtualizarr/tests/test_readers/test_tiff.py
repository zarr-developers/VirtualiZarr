import numpy as np
from xarray import DataArray

from virtualizarr import open_virtual_dataarray
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import requires_pillow


@requires_pillow
def test_random_tiff(random_tiff):
    vda = open_virtual_dataarray(random_tiff, indexes={})

    assert isinstance(vda, DataArray)

    assert vda.sizes == {"X": 128, "Y": 128}
    assert vda.dtype == np.uint8

    assert isinstance(vda.data, ManifestArray)
    manifest = vda.data.manifest
    assert manifest.dict() == {
        "0.0": {"path": random_tiff, "offset": 122, "length": 16384}
    }
