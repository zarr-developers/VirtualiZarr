import xarray as xr
from virtualizarr import open_virtual_dataset


def test_open_scalar_variable():
    # regression test for issue 100

    ds = xr.Dataset(data_vars={'a': 0})
    ds.to_netcdf('scalar.nc')

    vds = open_virtual_dataset('scalar.nc')
    assert vds['a'].shape == ()
