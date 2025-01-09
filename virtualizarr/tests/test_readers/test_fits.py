import pytest
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.tests import requires_kerchunk, requires_network

pytest.importorskip("astropy")


@requires_kerchunk
@requires_network
def test_open_hubble_data():
    # data from https://registry.opendata.aws/hst/
    vds = open_virtual_dataset(
        "s3://stpubdata/hst/public/f05i/f05i0201m/f05i0201m_a1f.fits",
        reader_options={"storage_options": {"anon": True}},
    )

    assert isinstance(vds, Dataset)
    assert list(vds.variables) == ["PRIMARY"]
    var = vds["PRIMARY"].variable
    assert var.sizes == {"y": 17, "x": 589}
    assert var.dtype == ">i4"
