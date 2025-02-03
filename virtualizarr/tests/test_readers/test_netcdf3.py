import numpy as np
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import requires_scipy
from virtualizarr.zarr import ZArray


@requires_scipy
def test_read_netcdf3(netcdf3_file):
    filepath = str(netcdf3_file)
    vds = open_virtual_dataset(filepath)

    assert isinstance(vds, xr.Dataset)
    assert list(vds.variables.keys()) == ["foo"]
    assert isinstance(vds["foo"].data, ManifestArray)

    expected_manifest = ChunkManifest(
        entries={"0": {"path": filepath, "offset": 80, "length": 12}}
    )
    expected_zarray = ZArray(dtype=np.dtype(">i4"), shape=(3,), chunks=(3,))
    expected_ma = ManifestArray(chunkmanifest=expected_manifest, zarray=expected_zarray)
    expected_vds = xr.Dataset({"foo": xr.Variable(data=expected_ma, dims=["x"])})

    xrt.assert_identical(vds, expected_vds)


# TODO test loading data against xarray backend, see issue #394 for context
