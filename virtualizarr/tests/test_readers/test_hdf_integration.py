import fsspec
import numpy
import pytest
import xarray as xr

import virtualizarr
from virtualizarr.kerchunk import FileType


class TestIntegration:
    def test_filters_h5netcdf_roundtrip(
        self, tmpdir, filter_encoded_xarray_h5netcdf_file
    ):
        virtual_ds = virtualizarr.open_virtual_dataset(
            filter_encoded_xarray_h5netcdf_file, filetype=FileType("netcdf4")
        )
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        virtual_ds.virtualize.to_kerchunk(kerchunk_file, format="json")
        fs = fsspec.filesystem("reference", fo=kerchunk_file)
        m = fs.get_mapper("")

        ds = xr.open_dataset(m, engine="kerchunk")
        assert isinstance(ds.air.values[0][0][0], numpy.float64)

    @pytest.mark.skip(
        reason="Issue with xr 'dim1' serialization and blosc availability"
    )
    def test_filters_netcdf4_roundtrip(
        self, tmpdir, filter_encoded_xarray_netcdf4_file
    ):
        filepath = filter_encoded_xarray_netcdf4_file["filepath"]
        compressor = filter_encoded_xarray_netcdf4_file["compressor"]
        virtual_ds = virtualizarr.open_virtual_dataset(
            filepath, filetype=FileType("netcdf4")
        )
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        virtual_ds.virtualize.to_kerchunk(kerchunk_file, format="json")
        fs = fsspec.filesystem("reference", fo=kerchunk_file)
        m = fs.get_mapper("")
        ds = xr.open_dataset(m, engine="kerchunk")

        expected_encoding = ds["var2"].encoding.copy()
        compression = expected_encoding.pop("compression")
        blosc_shuffle = expected_encoding.pop("blosc_shuffle")
        if compression is not None:
            if "blosc" in compression and blosc_shuffle:
                expected_encoding["blosc"] = {
                    "compressor": compressor,
                    "shuffle": blosc_shuffle,
                }
                expected_encoding["shuffle"] = False
        actual_encoding = ds["var2"].encoding
        assert expected_encoding.items() <= actual_encoding.items()
