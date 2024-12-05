# VirtualiZarr

**VirtualiZarr creates virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.**

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

## Motivation

The Kerchunk idea solves an incredibly important problem: accessing big archival datasets via a cloud-optimized pattern, but without copying or modifying the original data in any way. This is a win-win-win for users, data engineers, and data providers. Users see fast-opening zarr-compliant stores that work performantly with libraries like xarray and dask, data engineers can provide this speed by adding a lightweight virtualization layer on top of existing data (without having to ask anyone's permission), and data providers don't have to change anything about their legacy files for them to be used in a cloud-optimized way.

However, kerchunk's current design is limited:
- Store-level abstractions make combining datasets complicated, idiosyncratic, and requires duplicating logic that already exists in libraries like xarray,
- The kerchunk format for storing on-disk references requires the caller to understand it, usually via [`fsspec`](https://github.com/fsspec/filesystem_spec) (which is currently only implemented in python).

VirtualiZarr aims to build on the excellent ideas of kerchunk whilst solving the above problems:
- Using array-level abstractions instead is more modular, easier to reason about, allows convenient wrapping by high-level tools like xarray, and is simpler to parallelize,
- Writing the virtualized arrays out as a valid Zarr store directly (through new Zarr Extensions) will allow for Zarr implementations in any language to read the archival data.

## Aim

Let's say you have a bunch of legacy files (e.g. netCDF) which together tile along a dimension to form a large dataset. Let's imagine you already know how to use xarray to open these files and combine the opened dataset objects into one complete dataset. (If you don't then read the [xarray docs page on combining data](https://docs.xarray.dev/en/stable/user-guide/combining.html).)

```python
ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='h5netcdf',
    combine='nested',
)
ds  # the complete lazy xarray dataset
```

However, you don't want to run this set of xarray operations every time you open this dataset, as running commands like `xr.open_mfdataset` can be expensive. Instead you would prefer to just be able to open a virtualized Zarr store (i.e. `xr.open_dataset('my_virtual_store.zarr')`), as that would open instantly, but still give access to the same data underneath.

**`VirtualiZarr` aims to allow you to use the same xarray incantation you would normally use to open and combine all your files, but cache that result as a virtual Zarr store.**

What's being cached here, you ask? We're effectively caching the result of performing all the various consistency checks that xarray performs when it combines newly-encountered datasets together. Once you have the new virtual Zarr store xarray is able to assume that this checking has already been done, and trusts your Zarr store enough to just open it instantly.

### Usage

Creating the virtual store looks very similar to how we normally open data with xarray:

```python
from virtualizarr import open_virtual_dataset

virtual_datasets = [
    open_virtual_dataset(filepath)
    for filepath in glob.glob('/my/files*.nc')
]

# this Dataset wraps a bunch of virtual ManifestArray objects directly
virtual_ds = xr.combine_nested(virtual_datasets, concat_dim=['time'])

# cache the combined dataset pattern to disk, in this case using the existing kerchunk specification for reference files
virtual_ds.virtualize.to_kerchunk('combined.json', format='json')
```

Now you can open your shiny new Zarr store instantly:

```python
ds = xr.open_dataset('combined.json', engine='kerchunk', chunks={})  # normal xarray.Dataset object, wrapping dask/numpy arrays etc.
```

No data has been loaded or copied in this process, we have merely created an on-disk lookup table that points xarray into the specific parts of the original netCDF files when it needs to read each chunk.

See the [Usage docs page](#usage) for more details.

## Licence

Apache 2.0

## Site Contents

```{toctree}
:maxdepth: 2

self
installation
usage
examples
faq
api
releases
contributing
```
