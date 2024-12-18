# VirtualiZarr

**Create virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.**

The best way to distribute large scientific datasets is via the Cloud, in [Cloud-Optimized formats](https://guide.cloudnativegeo.org/) [^1]. But often this data is stuck in archival pre-Cloud file formats such as netCDF.

**VirtualiZarr[^2] makes it easy to create "Virtual" Zarr stores, allowing performant access to archival data as if it were in the Cloud-Optimized [Zarr format](https://zarr.dev/), _without duplicating any data_.**

## Motivation

"Virtualized data" solves an incredibly important problem: accessing big archival datasets via a cloud-optimized pattern, but without copying or modifying the original data in any way. This is a win-win-win for users, data engineers, and data providers. Users see fast-opening zarr-compliant stores that work performantly with libraries like xarray and dask, data engineers can provide this speed by adding a lightweight virtualization layer on top of existing data (without having to ask anyone's permission), and data providers don't have to change anything about their archival files for them to be used in a cloud-optimized way.

VirtualiZarr aims to make the creation of cloud-optimized virtualized zarr data from existing scientific data as easy as possible.

## Features

* Create virtual references pointing to bytes inside a archival file with [`open_virtual_dataset`](https://virtualizarr.readthedocs.io/en/latest/usage.html#opening-files-as-virtual-datasets),
* Supports a [range of archival file formats](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare), including netCDF4 and HDF5,
* [Combine data from multiple files](https://virtualizarr.readthedocs.io/en/latest/usage.html#combining-virtual-datasets) into one larger store using [xarray's combining functions](https://docs.xarray.dev/en/stable/user-guide/combining.html), such as [`xarray.concat`](https://docs.xarray.dev/en/stable/generated/xarray.concat.html),
* Commit the virtual references to storage either using the [Kerchunk references](https://fsspec.github.io/kerchunk/spec.html) specification or the [Icechunk](https://icechunk.io/) transactional storage engine.
* Users access the virtual dataset using [`xarray.open_dataset`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset).

## Inspired by Kerchunk

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [Kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk but in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

## Usage

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

## Talks and Presentations

- 2024/11/21 - MET Office Architecture Guild - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-talk-at-met-office)
- 2024/11/13 - Cloud-Native Geospatial conference - Raphael Hagen - [Slides](https://decks.carbonplan.org/cloud-native-geo/11-13-24)
- 2024/07/24 - ESIP Meeting - Sean Harkins - [Event](https://2024julyesipmeeting.sched.com/event/1eVP6) / [Recording](https://youtu.be/T6QAwJIwI3Q?t=3689)
- 2024/05/15 - Pangeo showcase - Tom Nicholas - [Event](https://discourse.pangeo.io/t/pangeo-showcase-virtualizarr-create-virtual-zarr-stores-using-xarray-syntax/4127/2) / [Recording](https://youtu.be/ioxgzhDaYiE) / [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-create-virtual-zarr-stores-using-xarray-syntax)

## Credits

This package was originally developed by [Tom Nicholas](https://github.com/TomNicholas) whilst working at [[C]Worthy](https://cworthy.org), who deserve credit for allowing him to prioritise a generalizable open-source solution to the dataset virtualization problem. VirtualiZarr is now a community-owned multi-stakeholder project.

## Licence

Apache 2.0

## Pages

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

## References

[^1]: [_Cloud-Native Repositories for Big Scientific Data_, Abernathey et. al., _Computing in Science & Engineering_.](https://ieeexplore.ieee.org/abstract/document/9354557)

[^2]: (Pronounced like "virtualizer" but more piratey 🦜)
