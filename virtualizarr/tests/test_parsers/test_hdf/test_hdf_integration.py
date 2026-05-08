import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_icechunk,
    requires_imagecodecs,
    requires_kerchunk,
)
from virtualizarr.tests.test_integration import roundtrip_as_in_memory_icechunk


@requires_kerchunk
@requires_hdf5plugin
@requires_imagecodecs
class TestIntegration:
    def test_filters_h5netcdf_roundtrip(
        self, tmp_path, filter_encoded_roundtrip_hdf5_file, local_registry
    ):
        url = f"file://{filter_encoded_roundtrip_hdf5_file}"
        parser = HDFParser()
        with (
            xr.open_dataset(
                filter_encoded_roundtrip_hdf5_file, decode_times=True
            ) as ds,
            open_virtual_dataset(
                url=url,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time"],
                decode_times=True,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.vz.to_kerchunk(kerchunk_file, format="json")
            with xr.open_dataset(
                kerchunk_file, engine="kerchunk", decode_times=True
            ) as roundtrip:
                xrt.assert_allclose(ds, roundtrip)

    def test_filters_netcdf4_roundtrip(
        self, tmp_path, filter_encoded_roundtrip_netcdf4_file, local_registry
    ):
        filepath = filter_encoded_roundtrip_netcdf4_file["filepath"]
        parser = HDFParser()
        with (
            xr.open_dataset(filepath) as ds,
            open_virtual_dataset(
                url=filepath,
                parser=parser,
                registry=local_registry,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.vz.to_kerchunk(kerchunk_file, format="json")
            with xr.open_dataset(kerchunk_file, engine="kerchunk") as roundtrip:
                xrt.assert_equal(ds, roundtrip)

    def test_filter_and_cf_roundtrip(
        self, tmp_path, filter_and_cf_roundtrip_hdf5_file, local_registry
    ):
        url = f"file://{filter_and_cf_roundtrip_hdf5_file}"
        parser = HDFParser()
        with (
            xr.open_dataset(filter_and_cf_roundtrip_hdf5_file) as ds,
            open_virtual_dataset(
                url=url,
                registry=local_registry,
                parser=parser,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "filter_cf_kerchunk.json")
            vds.vz.to_kerchunk(kerchunk_file, format="json")
            with xr.open_dataset(kerchunk_file, engine="kerchunk") as roundtrip:
                xrt.assert_allclose(ds, roundtrip)
                assert (
                    ds["temperature"].encoding["_FillValue"]
                    == roundtrip["temperature"].encoding["_FillValue"]
                )

    def test_non_coord_dim_roundtrip(self, tmp_path, non_coord_dim, local_registry):
        parser = HDFParser()
        with (
            xr.open_dataset(non_coord_dim) as ds,
            open_virtual_dataset(
                url=non_coord_dim,
                registry=local_registry,
                parser=parser,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.vz.to_kerchunk(kerchunk_file, format="json")
            with xr.open_dataset(kerchunk_file, engine="kerchunk") as roundtrip:
                xrt.assert_equal(ds, roundtrip)

    def test_singleton_dim_roundtrip(
        self, tmp_path, singleton_padded_dimension_hdf5_file, local_registry
    ):
        import numpy as np

        parser = HDFParser(squeeze=True)
        filepath, *_ = singleton_padded_dimension_hdf5_file
        with (
            xr.open_dataset(
                filepath, engine="h5netcdf", backend_kwargs={"phony_dims": "sort"}
            ).squeeze(drop=True) as ds,
            open_virtual_dataset(
                url=filepath,
                registry=local_registry,
                parser=parser,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.vz.to_kerchunk(kerchunk_file, format="json")
            with xr.open_dataset(kerchunk_file, engine="kerchunk") as roundtrip:
                for var_name in ds.data_vars:
                    assert ds[var_name].shape == roundtrip[var_name].shape
                    np.testing.assert_array_equal(
                        ds[var_name].values, roundtrip[var_name].values
                    )

    @requires_icechunk
    def test_cf_fill_value_roundtrip(
        self, tmp_path, cf_fill_value_hdf5_file, local_registry
    ):
        cf_fill_value_hdf5_url = f"file://{cf_fill_value_hdf5_file}"
        parser = HDFParser()
        with xr.open_dataset(cf_fill_value_hdf5_file, engine="h5netcdf") as ds:
            if ds["data"].dtype in [float, object]:
                pytest.xfail(
                    "TODO: fix handling fixed-length and structured type fill value"
                    " encoding in xarray zarr parser."
                )
            with open_virtual_dataset(
                url=cf_fill_value_hdf5_url,
                registry=local_registry,
                parser=parser,
            ) as vds:
                roundtrip = roundtrip_as_in_memory_icechunk(
                    vds, tmp_path, decode_times=False
                )
                xrt.assert_equal(ds, roundtrip)
