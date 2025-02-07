import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import requires_scipy


@requires_scipy
@pytest.mark.xfail(reason="zarr-python 3.0 does not support big endian")
def test_read_netcdf3(netcdf3_file, array_v3_metadata):
    filepath = str(netcdf3_file)
    vds = open_virtual_dataset(filepath)

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


# TODO test loading data against xarray backend, see issue #394 for context
