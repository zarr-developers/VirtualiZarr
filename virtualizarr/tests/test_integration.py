import xarray.testing as xrt
import xarray as xr
import fsspec

from virtualizarr import open_dataset_via_kerchunk


def test_kerchunk_roundtrip_no_concat(tmpdir):
    # set up example xarray dataset
    ds = xr.tutorial.open_dataset('air_temperature', decode_times=False)

    # save it to disk as netCDF (in temporary directory)
    ds.to_netcdf(f'{tmpdir}/air.nc')

    # use open_dataset_via_kerchunk to read it as references
    vds = open_dataset_via_kerchunk(f'{tmpdir}/air.nc', filetype='netCDF4')

    # write those references to disk as kerchunk json
    vds.virtualize.to_kerchunk(f'{tmpdir}/refs.json', format='json')

    # use fsspec to read the dataset from disk via the zarr store
    fs = fsspec.filesystem('reference', fo=f'{tmpdir}/refs.json')
    m = fs.get_mapper('')

    roundtrip = xr.open_dataset(m, engine='kerchunk')

    # assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)
