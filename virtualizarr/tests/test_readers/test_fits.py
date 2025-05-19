import pytest
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.backends import FITSBackend
from virtualizarr.tests import requires_kerchunk, requires_network
from virtualizarr.tests.utils import obstore_s3

pytest.importorskip("astropy")


@requires_kerchunk
@requires_network
@pytest.mark.xfail(
    reason="Big endian not yet supported by zarr-python 3.0"
)  # https://github.com/zarr-developers/zarr-python/issues/2324
def test_open_hubble_data():
    # data from https://registry.opendata.aws/hst/
    filepath = "s3://stpubdata/hst/public/f05i/f05i0201m/f05i0201m_a1f.fits"
    store = obstore_s3(filepath=filepath, region="us-west-2")
    backend = FITSBackend(reader_options={"storage_options": {"anon": True}})
    vds = open_virtual_dataset(
        filepath=filepath,
        object_reader=store,
        backend=backend,
    )

    assert isinstance(vds, Dataset)
    assert list(vds.variables) == ["PRIMARY"]
    var = vds["PRIMARY"].variable
    assert var.sizes == {"y": 17, "x": 589}
    assert var.dtype == ">i4"
