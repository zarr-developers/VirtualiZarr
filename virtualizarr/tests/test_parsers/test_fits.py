import pytest
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import FITSParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import requires_kerchunk, requires_network
from virtualizarr.tests.utils import obstore_s3

pytest.importorskip("astropy")


@requires_kerchunk
@requires_network
def test_open_hubble_data():
    # data from https://registry.opendata.aws/hst/
    url = "s3://stpubdata/hst/public/f05i/f05i0201m/f05i0201m_a1f.fits"
    store = obstore_s3(url=url, region="us-west-2")
    parser = FITSParser(reader_options={"storage_options": {"anon": True}})
    registry = ObjectStoreRegistry()
    registry.register(url, store)
    with open_virtual_dataset(
        url=url,
        registry=registry,
        parser=parser,
    ) as vds:
        assert isinstance(vds, Dataset)
        assert list(vds.variables) == ["PRIMARY"]
        var = vds["PRIMARY"].variable
        assert var.sizes == {"y": 17, "x": 589}
        assert var.dtype == "int32"
