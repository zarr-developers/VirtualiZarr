import numpy as np
import pytest
import xarray as xr
from obstore.store import LocalStore

from virtualizarr import open_virtual_dataset
from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.storage.obstore import ManifestStore
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
class TestManifestStore:
    def test_rountrip_simple_virtualdataset(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        vds = open_virtual_dataset(
            filepath,
            backend=HDFVirtualBackend,
        )
        stores = {"file://": LocalStore()}
        ms = ManifestStore(vds, stores=stores)
        rountripped_ds = xr.open_dataset(
            ms, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    # def test_convert_simple_dataset(self, tmpdir):
    #     "Pass a regular dataset into ManifestStore"
    #     raise NotImplementedError

    # def test_rountrip_virtualdataset_nans(self, tmpdir):
    #     "Roundtrip a dataset containing NaNs to/from NetCDF with the HDF reader and ManifestStore"
    #     raise NotImplementedError

    # def test_rountrip_virtualdataset_mask_and_scale(self, tmpdir):
    #     "Roundtrip a dataset with a scale and offset to/from NetCDF with the HDF reader and ManifestStore"
    #     raise NotImplementedError

    # def test_rountrip_virtualdataset_time(self, tmpdir):
    #     "Roundtrip a dataset containing datetime values to/from NetCDF with the HDF reader and ManifestStore"
    #     raise NotImplementedError
