import fsspec
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_dataset_via_kerchunk


def test_kerchunk_roundtrip_no_concat(tmpdir):
    # set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature", decode_times=False)

    # save it to disk as netCDF (in temporary directory)
    ds.to_netcdf(f"{tmpdir}/air.nc")

    # use open_dataset_via_kerchunk to read it as references
    vds = open_dataset_via_kerchunk(f"{tmpdir}/air.nc", filetype="netCDF4")

    # write those references to disk as kerchunk json
    vds.virtualize.to_kerchunk(f"{tmpdir}/refs.json", format="json")

    # use fsspec to read the dataset from disk via the zarr store
    fs = fsspec.filesystem("reference", fo=f"{tmpdir}/refs.json")
    m = fs.get_mapper("")

    roundtrip = xr.open_dataset(m, engine="kerchunk")

    # assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)


def test_kerchunk_roundtrip_concat(tmpdir):
    # set up example xarray dataset
    ds = xr.tutorial.open_dataset("air_temperature", decode_times=False).isel(time=slice(None, 2000))

    # split into two datasets
    ds1, ds2 = ds.isel(time=slice(None, 1000)), ds.isel(time=slice(1000, None))

    # save it to disk as netCDF (in temporary directory)
    ds1.to_netcdf(f"{tmpdir}/air1.nc")
    ds2.to_netcdf(f"{tmpdir}/air2.nc")


    # use open_dataset_via_kerchunk to read it as references
    vds1 = open_dataset_via_kerchunk(f"{tmpdir}/air1.nc", filetype="netCDF4")
    vds2 = open_dataset_via_kerchunk(f"{tmpdir}/air2.nc", filetype="netCDF4")

    # concatenate virtually along time
    vds = xr.concat([vds1, vds2], dim='time', coords='minimal', compat='override')
    print(vds['air'].variable._data)

    # write those references to disk as kerchunk json
    vds.virtualize.to_kerchunk(f"{tmpdir}/refs.json", format="json")

    # use fsspec to read the dataset from disk via the zarr store
    fs = fsspec.filesystem("reference", fo=f"{tmpdir}/refs.json")
    m = fs.get_mapper("")

    roundtrip = xr.open_dataset(m, engine="kerchunk")

    # user does analysis here

    # assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)