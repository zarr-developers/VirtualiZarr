import pytest
import xarray as xr

from virtualizarr import open_virtual_dataset
from virtualizarr.tests import network

urls = [
    (
        "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
    )
]


@network
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_dmrpp_reader(data_url, dmrpp_url):
    result = open_virtual_dataset(dmrpp_url, indexes={}, filetype="dmrpp")
    expected = open_virtual_dataset(data_url, indexes={})
    xr.testing.assert_identical(result, expected)
