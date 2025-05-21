from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import HDFParser
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_minio,
    requires_obstore,
)
from virtualizarr.tests.utils import obstore_local


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
class TestHDFManifestStore:
    def test_roundtrip_simple_virtualdataset(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        store = obstore_local(file_url=filepath)
        parser = HDFParser()
        manifest_store = parser(
            file_url=filepath,
            object_store=store,
        )
        rountripped_ds = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    def test_rountrip_partial_chunk_virtualdataset(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore with a single partial chunk"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        encoding = {
            "temperature": {"chunksizes": (90, 90), "original_shape": (100, 100)}
        }
        basic_ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
        store = obstore_local(file_url=filepath)
        parser = HDFParser()
        manifest_store = parser(
            file_url=filepath,
            object_store=store,
        )
        rountripped_ds = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    def test_rountrip_simple_virtualdataset_default_store(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        store = obstore_local(file_url=filepath)
        parser = HDFParser()
        manifest_store = parser(
            file_url=filepath,
            object_store=store,
        )
        rountripped_ds = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        xr.testing.assert_allclose(basic_ds, rountripped_ds)

    @requires_minio
    @requires_obstore
    def test_store(self, minio_bucket, chunked_roundtrip_hdf5_s3_file):
        import obstore as obs
        
        parsed = urlparse(chunked_roundtrip_hdf5_s3_file)
        path_without_file = str(Path(parsed.path).parent)
        parsed_without_file = parsed._replace(path=path_without_file)
        url_without_file = parsed_without_file.geturl()

        s3store = obs.store.from_url(
            url_without_file,
            config={
                "virtual_hosted_style_request": False,
                "skip_signature": True,
                "endpoint_url": "http://localhost:9000",
            },
            client_options={"allow_http": True},
        )
        parser = HDFParser()
        manifest_store = parser(
            file_url=chunked_roundtrip_hdf5_s3_file, object_store=s3store
        )

        vds = manifest_store.to_virtual_dataset()
        assert vds.dims == {"phony_dim_0": 5}
        assert isinstance(vds["data"].data, ManifestArray)
