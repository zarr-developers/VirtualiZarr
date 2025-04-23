import numpy as np
import pytest
import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_minio,
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
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        store = HDFVirtualBackend._create_manifest_store(
            uri=filepath,
        )
        rountripped_ds = xr.open_dataset(
            store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    def test_rountrip_simple_virtualdataset_default_store(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        store = HDFVirtualBackend._create_manifest_store(uri=filepath)
        rountripped_ds = xr.open_dataset(
            store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    @requires_minio
    @requires_obstore
    def test_store(self, minio_bucket, chunked_roundtrip_hdf5_s3_file):
        import obstore as obs

        s3store = obs.store.S3Store(
            bucket=minio_bucket["bucket"],
            config={
                "endpoint": minio_bucket["endpoint"],
                "virtual_hosted_style_request": False,
                "skip_signature": True,
            },
            client_options={"allow_http": True},
        )
        store = HDFVirtualBackend._create_manifest_store(
            uri=chunked_roundtrip_hdf5_s3_file,
            store=s3store,
        )
        vds = store.to_virtual_dataset()
        assert vds.dims == {"phony_dim_0": 5}
        assert isinstance(vds["data"].data, ManifestArray)

    @requires_obstore
    def test_default_store(self):
        store = HDFVirtualBackend._create_manifest_store(
            uri="s3://carbonplan-share/virtualizarr/local.nc",
        )
        vds = store.to_virtual_dataset()
        assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}
        assert isinstance(vds["air"].data, ManifestArray)
        for name in ["time", "lat", "lon"]:
            assert isinstance(vds[name].data, np.ndarray)
