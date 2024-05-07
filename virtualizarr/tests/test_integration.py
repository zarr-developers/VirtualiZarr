import xarray as xr
from virtualizarr import open_virtual_dataset


def test_open_scalar_variable(tmpdir):
    # regression test for GH issue #100

    ds = xr.Dataset(data_vars={'a': 0})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")

    vds = open_virtual_dataset(f"{tmpdir}/scalar.nc")
    assert vds['a'].shape == ()
