import pytest
import xarray as xr
import xarray.testing as xrt

import virtualizarr
from virtualizarr.kerchunk import FileType


class TestIntegration:
    @pytest.mark.xfail(reason="Investigate initial time value decoding issue")
    def test_filters_h5netcdf_roundtrip(
        self, tmpdir, filter_encoded_xarray_h5netcdf_file
    ):
        ds = xr.open_dataset(filter_encoded_xarray_h5netcdf_file, decode_times=False)
        vds = virtualizarr.open_virtual_dataset(
            filter_encoded_xarray_h5netcdf_file, filetype=FileType("netcdf4")
        )
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        vds.virtualize.to_kerchunk(kerchunk_file, format="json")
        roundtrip = xr.open_dataset(
            kerchunk_file, engine="kerchunk", decode_times=False
        )
        xrt.assert_allclose(ds, roundtrip)

    def test_filters_netcdf4_roundtrip(
        self, tmpdir, filter_encoded_xarray_netcdf4_file
    ):
        filepath = filter_encoded_xarray_netcdf4_file["filepath"]
        ds = xr.open_dataset(filepath)
        vds = virtualizarr.open_virtual_dataset(filepath, filetype=FileType("netcdf4"))
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        vds.virtualize.to_kerchunk(kerchunk_file, format="json")
        roundtrip = xr.open_dataset(kerchunk_file, engine="kerchunk")
        xrt.assert_equal(ds, roundtrip)
