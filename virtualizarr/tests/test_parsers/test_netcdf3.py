import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import NetCDF3Parser
from virtualizarr.tests import requires_network, requires_scipy
from virtualizarr.tests.utils import obstore_http, obstore_local


@requires_scipy
def test_read_netcdf3(netcdf3_file, array_v3_metadata):
    filepath = str(netcdf3_file)
    store = obstore_local(file_url=filepath)
    parser = NetCDF3Parser()
    with (
        parser(file_url=filepath, object_store=store) as manifest_store,
        xr.open_dataset(filepath) as expected,
    ):
        observed = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        assert isinstance(observed, xr.Dataset)
        assert list(observed.variables.keys()) == ["foo"]
        xrt.assert_identical(observed.load(), expected.load())


@requires_network
def test_read_http_netcdf3(array_v3_metadata):
    file_url = "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc"
    store = obstore_http(file_url=file_url)
    parser = NetCDF3Parser()
    with open_virtual_dataset(
        file_url=file_url,
        parser=parser,
        object_store=store,
    ) as vds:
        assert isinstance(vds, xr.Dataset)


# TODO test loading data against xarray backend, see issue #394 for context
