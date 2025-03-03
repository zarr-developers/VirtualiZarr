import pytest
import xarray as xr
import xarray.testing as xrt

import virtualizarr
from virtualizarr.readers.hdf import HDFVirtualBackend
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
    @pytest.mark.xfail(
        reason="0 time start is being interpreted as fillvalue see issues/280"
    )
    def test_filters_h5netcdf_roundtrip(
        self, tmp_path, filter_encoded_roundtrip_hdf5_file
    ):
        with (
            xr.open_dataset(
                filter_encoded_roundtrip_hdf5_file, decode_times=True
            ) as ds,
            virtualizarr.open_virtual_dataset(
                filter_encoded_roundtrip_hdf5_file,
                loadable_variables=["time"],
                cftime_variables=["time"],
                backend=HDFVirtualBackend,
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.virtualize.to_kerchunk(kerchunk_file, format="json")
            roundtrip = xr.open_dataset(
                kerchunk_file, engine="kerchunk", decode_times=True
            )
            xrt.assert_allclose(ds, roundtrip)

    def test_filters_netcdf4_roundtrip(
        self, tmp_path, filter_encoded_roundtrip_netcdf4_file
    ):
        filepath = filter_encoded_roundtrip_netcdf4_file["filepath"]
        with (
            xr.open_dataset(filepath) as ds,
            virtualizarr.open_virtual_dataset(
                filepath, backend=HDFVirtualBackend
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.virtualize.to_kerchunk(kerchunk_file, format="json")
            roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
            xrt.assert_equal(ds, roundtrip)

    def test_filter_and_cf_roundtrip(self, tmp_path, filter_and_cf_roundtrip_hdf5_file):
        with (
            xr.open_dataset(filter_and_cf_roundtrip_hdf5_file) as ds,
            virtualizarr.open_virtual_dataset(
                filter_and_cf_roundtrip_hdf5_file, backend=HDFVirtualBackend
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "filter_cf_kerchunk.json")
            vds.virtualize.to_kerchunk(kerchunk_file, format="json")
            roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
            xrt.assert_allclose(ds, roundtrip)
            assert (
                ds["temperature"].encoding["_FillValue"]
                == roundtrip["temperature"].encoding["_FillValue"]
            )

    def test_non_coord_dim_roundtrip(self, tmp_path, non_coord_dim):
        with (
            xr.open_dataset(non_coord_dim) as ds,
            virtualizarr.open_virtual_dataset(
                non_coord_dim, backend=HDFVirtualBackend
            ) as vds,
        ):
            kerchunk_file = str(tmp_path / "kerchunk.json")
            vds.virtualize.to_kerchunk(kerchunk_file, format="json")
            roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
            xrt.assert_equal(ds, roundtrip)

    @requires_icechunk
    def test_cf_fill_value_roundtrip(self, tmp_path, cf_fill_value_hdf5_file):
        with xr.open_dataset(cf_fill_value_hdf5_file, engine="h5netcdf") as ds:
            if ds["data"].dtype in [float, object]:
                pytest.xfail(
                    "TODO: fix handling fixed-length and structured type fill value"
                    " encoding in xarray zarr backend."
                )
            with virtualizarr.open_virtual_dataset(
                cf_fill_value_hdf5_file, backend=HDFVirtualBackend
            ) as vds:
                roundtrip = roundtrip_as_in_memory_icechunk(
                    vds, tmp_path, decode_times=False
                )
                xrt.assert_equal(ds, roundtrip)
