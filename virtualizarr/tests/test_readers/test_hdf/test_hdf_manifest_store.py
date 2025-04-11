import numpy as np
import pytest
import xarray as xr

from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_obstore,
)


@pytest.fixture(name="basic_ds")
def basic_ds():
    x = np.arange(100)
    y = np.arange(100)
    temperature = 0.1 * x[:, None] + 0.1 * y[None, :]
    ds = xr.Dataset(
        {"temperature": (["x", "y"], temperature)},
        coords={"x": np.arange(100), "y": np.arange(100)},
    )
    return ds


@requires_hdf5plugin
@requires_obstore
class TestHDFManifestStore:
    def test_rountrip_simple_virtualdataset(self, tmpdir, basic_ds):
        from obstore.store import LocalStore

        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        store = HDFVirtualBackend._create_manifest_store(
            filepath=filepath, store=LocalStore(), prefix="file://"
        )
        rountripped_ds = xr.open_dataset(
            store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)
