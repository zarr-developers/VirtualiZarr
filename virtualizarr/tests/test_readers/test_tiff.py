import numpy as np
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.tests import requires_pillow


@requires_pillow
def test_random_tiff(random_tiff):
    vds = open_virtual_dataset(random_tiff, indexes={})

    assert isinstance(vds, Dataset)

    # TODO what is the name of this array expected to be??
    assert list(vds.variables) == ["foo"]
    vda = vds["foo"]

    assert vda.sizes == {"X": 128, "Y": 128}
    assert vda.dtype == np.uint8

    assert isinstance(vda.data, ManifestArray)
    manifest = vda.data.manifest
    assert manifest.dict() == {
        "0.0": {"path": random_tiff, "offset": 122, "length": 16384}
    }
