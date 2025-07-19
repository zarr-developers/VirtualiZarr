import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import requires_kerchunk, requires_network, requires_scipy
from virtualizarr.tests.utils import obstore_http


@requires_scipy
def test_read_netcdf3(netcdf3_file, array_v3_metadata, local_registry):
    filepath = str(netcdf3_file)
    url = f"file://{filepath}"
    parser = NetCDF3Parser()
    with (
        parser(url=url, registry=local_registry) as manifest_store,
        xr.open_dataset(filepath) as expected,
    ):
        observed = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        assert isinstance(observed, xr.Dataset)
        assert list(observed.variables.keys()) == ["foo"]
        xrt.assert_identical(observed.load(), expected.load())


@requires_kerchunk
@requires_network
def test_read_http_netcdf3(array_v3_metadata):
    url = "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc"
    store = obstore_http(url=url)
    registry = ObjectStoreRegistry({url: store})
    parser = NetCDF3Parser()
    with open_virtual_dataset(
        url=url,
        parser=parser,
        registry=registry,
    ) as vds:
        assert isinstance(vds, xr.Dataset)


# TODO test loading data against xarray backend, see issue #394 for context
