# VirtualiZarr

## Create virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.

The best way to distribute large scientific datasets is via the Cloud, in [Cloud-Optimized formats](https://guide.cloudnativegeo.org/) [^1]. But often this data is stuck in legacy pre-Cloud file formats such as netCDF.

**VirtualiZarr[^2] makes it easy to create "Virtual" Zarr stores, allowing performant access to legacy data as if it were in the Cloud-Optimized [Zarr format](https://zarr.dev/), _without duplicating any data_.**

[^1]: [_Cloud-Native Repositories for Big Scientific Data_, Abernathey et. al., _Computing in Science & Engineering_.](https://ieeexplore.ieee.org/abstract/document/9354557)

[^2]: (Pronounced like "virtualizer" but more piratey 🦜)

### Motivation

"Virtualized data" solves an incredibly important problem: accessing big archival datasets via a cloud-optimized pattern, but without copying or modifying the original data in any way. This is a win-win-win for users, data engineers, and data providers. Users see fast-opening zarr-compliant stores that work performantly with libraries like xarray and dask, data engineers can provide this speed by adding a lightweight virtualization layer on top of existing data (without having to ask anyone's permission), and data providers don't have to change anything about their legacy files for them to be used in a cloud-optimized way.

VirtualiZarr aims to make the creation of cloud-optimized virtualized zarr data from existing scientific data as easy as possible. 

### Features

* Create virtual references pointing to bytes inside a legacy file with [`open_virtual_dataset`](https://virtualizarr.readthedocs.io/en/latest/usage.html#opening-files-as-virtual-datasets),
* Supports a [range of legacy file formats](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare), including netCDF4 and HDF5,
* Combine the data from multiple files into one larger store using [simple functions like `xarray.concat`](https://virtualizarr.readthedocs.io/en/latest/usage.html#combining-virtual-datasets),
* Commit the virtual references to storage either using the [Kerchunk references](https://fsspec.github.io/kerchunk/spec.html) specification or the [Icechunk](https://icechunk.io/) transactional storage engine.
* Users access the virtual dataset using [`xarray.open_dataset`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset).

### VirtualiZarr vs Kerchunk?

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [Kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

### Xarray

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
