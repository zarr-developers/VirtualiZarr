import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.parsers import NetCDF3Parser
from virtualizarr.tests import requires_network, requires_scipy
from virtualizarr.tests.utils import obstore_http, obstore_local


@requires_scipy
@pytest.mark.xfail(
    reason="Big endian not yet supported by zarr-python 3.0"
)  # https://github.com/zarr-developers/zarr-python/issues/2324
def test_read_netcdf3(netcdf3_file, array_v3_metadata):
    filepath = str(netcdf3_file)
    store = obstore_local(file_url=filepath)
    parser = NetCDF3Parser()
    with open_virtual_dataset(
        file_url=filepath,
        parser=parser,
        object_store=store,
    ) as vds:
        assert isinstance(vds, xr.Dataset)
        assert list(vds.variables.keys()) == ["foo"]
        assert isinstance(vds["foo"].data, ManifestArray)

        expected_manifest = ChunkManifest(
            entries={"0": {"path": filepath, "offset": 80, "length": 12}}
        )
        metadata = array_v3_metadata(shape=(3,), chunks=(3,))
        expected_ma = ManifestArray(chunkmanifest=expected_manifest, metadata=metadata)
        expected_vds = xr.Dataset({"foo": xr.Variable(data=expected_ma, dims=["x"])})

        xrt.assert_identical(vds, expected_vds)


@requires_network
@pytest.mark.xfail(
    reason="Big endian not yet supported by zarr-python 3.0"
)  # https://github.com/zarr-developers/zarr-python/issues/2324
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
