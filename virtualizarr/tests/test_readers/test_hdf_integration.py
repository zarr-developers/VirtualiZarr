import fsspec
import numpy
import xarray as xr

import virtualizarr
from virtualizarr.kerchunk import FileType


class TestIntegration:
    def test_filters_end_to_end(self, tmpdir,
                                    filter_encoded_xarray_netcdf4_files):
        virtual_ds = virtualizarr.open_virtual_dataset(
            filter_encoded_xarray_netcdf4_files,
            filetype=FileType("netcdf4")
        )
        kerchunk_file = f"{tmpdir}/kerchunk.json"
        virtual_ds.virtualize.to_kerchunk(kerchunk_file, format="json")
        fs = fsspec.filesystem("reference", fo=kerchunk_file)
        m = fs.get_mapper("")

        ds = xr.open_dataset(m, engine="kerchunk")
        assert isinstance(ds.air.values[0][0][0], numpy.float64)
