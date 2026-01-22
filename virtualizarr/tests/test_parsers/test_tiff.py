import pytest
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import S3Store
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.tests import requires_network, requires_tiff

virtual_tiff = pytest.importorskip("virtual_tiff")


@requires_tiff
@requires_network
def test_virtual_tiff() -> None:
    store = S3Store("sentinel-cogs", region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({"s3://sentinel-cogs/": store})
    url = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/12/S/UF/2022/6/S2B_12SUF_20220609_0_L2A/B04.tif"
    parser = virtual_tiff.VirtualTIFF(ifd=0)
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, Dataset)
        assert list(vds.variables) == ["0"]
        var = vds["0"].variable
        assert var.sizes == {"y": 10980, "x": 10980}
        assert var.dtype == "<u2"
