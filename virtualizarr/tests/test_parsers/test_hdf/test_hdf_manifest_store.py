from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_minio,
    requires_obstore,
)
from virtualizarr.tests.utils import manifest_store_from_hdf_url


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
        url = f"file://{filepath}"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        manifest_store = manifest_store_from_hdf_url(url)
        with xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        ) as rountripped_ds:
            xr.testing.assert_allclose(basic_ds, rountripped_ds)

    def test_rountrip_partial_chunk_virtualdataset(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore with a single partial chunk"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        url = f"file://{filepath}"
        encoding = {
            "temperature": {"chunksizes": (90, 90), "original_shape": (100, 100)}
        }
        basic_ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
        manifest_store = manifest_store_from_hdf_url(url)
        with xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        ) as rountripped_ds:
            xr.testing.assert_allclose(basic_ds, rountripped_ds)

    def test_rountrip_simple_virtualdataset_default_store(self, tmpdir, basic_ds):
        "Roundtrip a dataset to/from NetCDF with the HDF reader and ManifestStore"

        filepath = f"{tmpdir}/basic_ds_roundtrip.nc"
        url = f"file://{filepath}"
        basic_ds.to_netcdf(filepath, engine="h5netcdf")
        manifest_store = manifest_store_from_hdf_url(url)
        with xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        ) as rountripped_ds:
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
        registry = ObjectStoreRegistry()
        registry.register(url_without_file, s3store)
        parser = HDFParser()
        manifest_store = parser(url=chunked_roundtrip_hdf5_s3_file, registry=registry)

        with manifest_store.to_virtual_dataset() as vds:
            assert vds.dims == {"phony_dim_0": 5}
            assert isinstance(vds["data"].data, ManifestArray)
            with xr.open_dataset(
                manifest_store, engine="zarr", consolidated=False, zarr_format=3
            ) as ds:
                assert ds.load()
