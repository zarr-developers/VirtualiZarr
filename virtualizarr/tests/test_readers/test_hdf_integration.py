import pytest
import xarray as xr
import xarray.testing as xrt

import virtualizarr
from virtualizarr.kerchunk import FileType


class TestIntegration:
    def test_filters_h5netcdf_roundtrip(
        self, tmpdir, filter_encoded_roundtrip_hdf5_file
    ):
        ds = xr.open_dataset(filter_encoded_roundtrip_hdf5_file, decode_times=True)
        vds = virtualizarr.open_virtual_dataset(
            filter_encoded_roundtrip_hdf5_file,
            loadable_variables=["time"],
            cftime_variables=["time"],
        )
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        vds.virtualize.to_kerchunk(kerchunk_file, format="json")
        roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk", decode_times=True)
        xrt.assert_allclose(ds, roundtrip)

    def test_filters_netcdf4_roundtrip(
        self, tmpdir, filter_encoded_roundtrip_netcdf4_file
    ):
        filepath = filter_encoded_roundtrip_netcdf4_file["filepath"]
        ds = xr.open_dataset(filepath)
        vds = virtualizarr.open_virtual_dataset(filepath, filetype=FileType("netcdf4"))
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        vds.virtualize.to_kerchunk(kerchunk_file, format="json")
        roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
        xrt.assert_equal(ds, roundtrip)

    @pytest.mark.xfail(reason="Investigate kerchunk _FillValue logic")
    def test_filter_and_cf_roundtrip(self, tmpdir, filter_and_cf_roundtrip_hdf5_file):
        ds = xr.open_dataset(filter_and_cf_roundtrip_hdf5_file)
        vds = virtualizarr.open_virtual_dataset(filter_and_cf_roundtrip_hdf5_file)
        kerchunk_file = f"{tmpdir}/filter_cf_kerchunk.json"
        vds.virtualize.to_kerchunk(kerchunk_file, format="json")
        roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
        xrt.assert_allclose(ds, roundtrip)
