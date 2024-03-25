# Usage

**NOTE: This package is in development. The usage examples in this section are currently aspirational. Progress towards making these examples work is tracked in [issue #2](https://github.com/TomNicholas/VirtualiZarr/issues/2).**

Let's say you have a bunch of legacy files (e.g. netCDF) which together tile to form a large dataset. Let's imagine you already know how to use xarray to open these files and combine the opened dataset objects into one complete dataset. (If you don't then read the [xarray docs page on combining data](https://docs.xarray.dev/en/stable/user-guide/combining.html).)

```python
ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='h5netcdf',
    combine='by_coords',  # 'by_coords' requires reading coord data to determine concatenation order
)
ds  # the complete lazy xarray dataset
```

However, you don't want to run this set of xarray operations every time you open this dataset, as running commands like `xr.open_mfdataset` can be expensive. Instead you would prefer to just be able to open a virtualized Zarr store (i.e. `xr.open_dataset('my_virtual_store.zarr')`), as that would open instantly, but still give access to the same data underneath.

**`VirtualiZarr` allows you to use the same xarray incantation you would normally use to open and combine all your files, but cache that result as a virtual Zarr store.**

What's being cached here, you ask? We're effectively caching the result of performing all the various consistency checks that xarray performs when it combines newly-encountered datasets together. Once you have the new virtual Zarr store xarray is able to assume that this checking has already been done, and trusts your Zarr store enough to just open it instantly.

Creating the virtual store looks very similar to how we normally open data with xarray:

```python
import virtualizarr  # required for the xarray backend and accessor to be present

virtual_ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='virtualizarr',  # virtualizarr registers an xarray IO backend that returns ManifestArray objects
    combine='by_coords',  # 'by_coords' stills requires actually reading coordinate data
)

virtual_ds  # now wraps a bunch of virtual ManifestArray objects directly

# cache the combined dataset pattern to disk, in this case using the existing kerchunk specification for reference files
virtual_ds.virtualize.to_kerchunk('combined.json', format='json')
```

Now you can open your shiny new Zarr store instantly:

```python
fs = fsspec.filesystem('reference', fo='combined.json')
m = fs.get_mapper('')

ds = xr.open_dataset(m, engine='kerchunk', chunks={})  # normal xarray.Dataset object, wrapping dask/numpy arrays etc.
```

(Since we serialized the cached results using the kerchunk specification then opening this zarr store still requires using fsspec via the kerchunk xarray backend.)

No data has been loaded or copied in this process, we have merely created an on-disk lookup table that points xarray into the specific parts of the original netCDF files when it needs to read each chunk.
