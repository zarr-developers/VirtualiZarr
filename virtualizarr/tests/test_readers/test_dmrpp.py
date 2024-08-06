import pytest
import xarray as xr

from virtualizarr import open_virtual_dataset
from virtualizarr.tests import network

urls = [
    (
        "netcdf4",
        "https://github.com/OPENDAP/bes/raw/3e518f6dc2f625b0b83cfb6e6fd5275e4d6dcef1/modules/dmrpp_module/data/dmrpp/chunked_threeD.h5",
        "dmrpp",
        "https://github.com/OPENDAP/bes/raw/3e518f6dc2f625b0b83cfb6e6fd5275e4d6dcef1/modules/dmrpp_module/data/dmrpp/chunked_threeD.h5.dmrpp",
    )
]


@network
@pytest.mark.parametrize("data_type, data_url, dmrpp_type, dmrpp_url", urls)
def test_dmrpp_reader(data_type, data_url, dmrpp_type, dmrpp_url):
    result = open_virtual_dataset(dmrpp_url, indexes={}, filetype=dmrpp_type)
    expected = open_virtual_dataset(data_url, indexes={})
    xr.testing.assert_identical(result, expected)
