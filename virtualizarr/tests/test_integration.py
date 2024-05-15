import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset


def test_open_scalar_variable(tmpdir):
    # regression test for GH issue #100

    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")

    vds = open_virtual_dataset(f"{tmpdir}/scalar.nc")
    assert vds["a"].shape == ()


@pytest.mark.parametrize("format", ["json", "parquet"])
def test_kerchunk_roundtrip(tmpdir, format):
    # set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature", decode_times=False)

    # save it to disk as netCDF (in temporary directory)
    ds.to_netcdf(f"{tmpdir}/air.nc")

    # use open_virtual_dataset to read it as references
    vds = open_virtual_dataset(f"{tmpdir}/air.nc", indexes={})

    # write those references to disk as kerchunk json
    vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

    # read the dataset from disk via the zarr store
    roundtrip = xr.open_dataset(f"{tmpdir}/refs.{format}", engine="kerchunk")

    # assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)
