import pytest
import xarray as xr

from virtualizarr import open_virtual_dataset
from virtualizarr.kerchunk import FileType

urls = [
    (
        "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/MUR25-JPL-L4-GLOB-v04.2/20240713090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
        "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/MUR25-JPL-L4-GLOB-v04.2/20240713090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "netcdf4",
    )
]


def match_zlib_level(result: xr.Dataset, expected: xr.Dataset):
    # Fix the zlib level in the result to match the expected
    # Many dmrpp's currently do not have the zlib level in the metadata (so default of -1 is used)
    # Usage of the virtual Dataset is not affected, but the comparison will fail
    # Remove once NASA dmrpps are updated with new dmrpp version: https://github.com/OPENDAP/bes/issues/954
    for x in result.variables:
        if result[x].data.zarray.dict()["filters"] is not None:
            e_filters = [
                z
                for z in expected[x].data.zarray.dict()["filters"]
                if "id" in z and z["id"] == "zlib"
            ][0]
            r_filters = [
                z
                for z in result[x].data.zarray.dict()["filters"]
                if "id" in z and z["id"] == "zlib"
            ][0]
            r_filters["level"] = e_filters["level"]


@pytest.mark.parametrize("dmrpp_url, data_url, data_type", urls)
def test_dmrpp_reader(dmrpp_url: str, data_url: str, data_type: str):
    import earthaccess

    fs = earthaccess.get_fsspec_https_session()
    result = open_virtual_dataset(
        dmrpp_url,
        indexes={},
        filetype=FileType("dmrpp"),
        reader_options={"storage_options": fs.storage_options},
    )
    expected = open_virtual_dataset(
        data_url,
        indexes={},
        filetype=FileType(data_type),
        reader_options={"storage_options": fs.storage_options},
    )
    match_zlib_level(result, expected)
    xr.testing.assert_identical(result, expected)
