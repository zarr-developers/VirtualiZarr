# VirtualiZarr

**Create virtual Zarr stores for cloud-friendly access to netCDF, HDF5, GRIB, TIFF and other formats, using familiar Xarray syntax.**

VirtualiZarr does three things.

1. **Assembles many files into a hypercube**, combining them into one dataset and checking that the result is valid Zarr.
2. **Reads files on the fly** as though they were Zarr, using zarr-python or Xarray.
3. **Persists the result to Icechunk**, so anyone can open it with Zarr or Xarray from then on.

See [How it works](#how-it-works) for more on each.

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

## How it works

### Assembling a hypercube

A parser reads each file and maps it onto Zarr: its arrays, its metadata, and where every chunk lives.
VirtualiZarr has parsers for [many formats](explanation/faq.md#can-my-file-format-be-virtualized).
You then combine the files into one dataset using [Xarray's combining logic](how_to/usage.md#combining-virtual-datasets), which matches variables and dimensions by name and checks that the files line up.
On top of that, VirtualiZarr refuses combinations that Zarr can't represent, such as files with different codecs, data types or chunk shapes, rather than producing references that would read back wrong.

### Reading on the fly

Some files are already cloud-optimized, such as cloud-optimized GeoTIFFs, so they don't need rewriting, but your tools may only work with Zarr.
When VirtualiZarr parses a file, it creates a Zarr store that reads from that file, so zarr-python and Xarray can load its data directly, without persisting anything first (see [Reading data from the `ManifestStore`](explanation/custom_parsers.md#reading-data-from-the-manifeststore)).

### Persisting to Icechunk

Writing the combined dataset to [Icechunk](https://icechunk.io/) lets you, or anyone else, reopen it later with zarr-python, without VirtualiZarr or Xarray in the read path.
Xarray users can open it with [xarray.open_zarr][].
The work of parsing and assembling the dataset only has to happen once, and every later read benefits from it (see [Writing to an Icechunk Store](how_to/usage.md#writing-to-an-icechunk-store)).

## Inspired by Kerchunk

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [Kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk but in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

## Quick usage example

Creating the virtual dataset looks quite similar to how we normally open data with [xarray][], but there are a few notable differences that are shown through this example.

First, import the necessary functions and classes:

```python exec="on" source="above" session="homepage"
import icechunk
import obstore

from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import (
    open_virtual_dataset,
    open_virtual_mfdataset,
)
from virtualizarr.parsers import HDFParser
```

```python exec="on" session="homepage"
# This code isn't shown since we didn't set source="above"
import xarray as xr
xr.set_options(display_style="html")
```

We can use Obstore's [`obstore.store.from_url`][obstore.store.from_url] convenience method to create an [ObjectStore][obstore.store.ObjectStore] that can fetch the data needed to virtualize the file.
The store holds the settings for connecting to the bucket, such as its cloud region and credentials.
This bucket is public and in `us-west-2`, so we set the region and skip signing requests.

```python exec="on" source="above" session="homepage"
bucket = "s3://nex-gddp-cmip6"
path = (
    "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/"
    "tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
)
store = obstore.store.from_url(
    bucket, region="us-west-2", skip_signature=True
)
```

A virtual dataset can pull from several sources, such as different buckets, different clouds, or HTTPS websites, and each source needs its own store.
An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] organizes those stores for VirtualiZarr by mapping each URL prefix to the store that serves it.
Here there is only one source, so the registry maps the bucket to our store.
See [The object store registry](explanation/registry.md) for more on why it's needed.

```python exec="on" source="above" session="homepage"
registry = ObjectStoreRegistry({bucket: store})
```

Now, let's create a parser instance and create a virtual dataset by passing the URL, parser, and registry to [virtualizarr.open_virtual_dataset][].

```python exec="on" source="above" session="homepage" result="code"
parser = HDFParser()
vds = open_virtual_dataset(
  url=f"{bucket}/{path}",
  parser=parser,
  registry=registry,
  loadable_variables=[],
)
print(vds)
```

Since we specified `loadable_variables=[]`, no data has been loaded or copied in this process. We have merely created an in-memory lookup table that points to the location of chunks in the original netCDF when data is needed later on. The default behavior (`loadable_variables=None`) will load data associated with coordinates but not data variables. The size represents the size of the original dataset - you can see the size of the virtual dataset using the `vz` accessor:

```python exec="on" source="above" session="homepage" result="code"
print(f"Original dataset size: {vds.nbytes} bytes")
print(f"Virtual dataset size: {vds.vz.nbytes} bytes")
```

VirtualiZarr's other top-level function is [virtualizarr.open_virtual_mfdataset][], which can open and virtualize multiple data sources into
a single virtual dataset, similar to how [xarray.open_mfdataset][] opens multiple data files as a single dataset.

```python exec="on" source="above" session="homepage" result="code"
urls = [
    f"{bucket}/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/"
    f"tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_{year}_v2.0.nc"
    for year in range(2015, 2017)
]
vds = open_virtual_mfdataset(
    urls, parser=parser, registry=registry
)
print(vds)
```

The magic of VirtualiZarr is that you can persist the virtual dataset to disk in a chunk references format such as [Icechunk](https://icechunk.io/),
meaning that the work of constructing the single coherent dataset only needs to happen once.
For subsequent data access, you can use [xarray.open_zarr][] to open that Icechunk store, which on object storage is
far faster than using [xarray.open_mfdataset][] to open the the original non-cloud-optimized files.

Let's persist the Virtual dataset using Icechunk. First let's create an Icechunk configuration with permissions to access our data.

```python exec="on" source="above" session="homepage"
config = icechunk.RepositoryConfig.default()
container = icechunk.VirtualChunkContainer(
    url_prefix="s3://nex-gddp-cmip6/",
    store=icechunk.s3_store(region="us-west-2", anonymous=True),
)
config.set_virtual_chunk_container(container)
```

Now we can store the references to our data. Here we store the references in an icechunk store that only lives in memory, but in most cases you'll store the "virtual" icechunk store in the cloud.

```python exec="on" source="above" session="homepage"
icechunk_store = icechunk.in_memory_storage()
repo = icechunk.Repository.create(icechunk_store, config)
session = repo.writable_session("main")
vds.vz.to_icechunk(session.store)
session.commit("Create virtual store")
```

See the [Usage docs page](how_to/usage.md) for more details.

## Articles

- 2026/07/13 - Earthmover blog - Tom Nicholas and Matt Iannucci - [Virtually Gribberish - Bringing Icechunk clarity to GRIB archives](https://www.earthmover.io/blog/virtual-grib-nbm)
- 2026/06/02 - Earthmover blog - Tom Nicholas - [Old format, no problem!: Cloud-optimizing the GOES-16 archive as Virtual Zarr](https://www.earthmover.io/blog/virtual-zarr)

## Talks and Presentations

- 2025/04/30 - Cloud-Native Geospatial Forum - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-and-icechunk-build-a-cloud-optimized-datacube-in-3-lines) / [Recording](https://youtu.be/QBkZQ53vE6o)
- 2024/11/21 - MET Office Architecture Guild - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-talk-at-met-office)
- 2024/11/13 - Cloud-Native Geospatial conference - Raphael Hagen - [Slides](https://decks.carbonplan.org/cloud-native-geo/11-13-24)
- 2024/07/24 - ESIP Meeting - Sean Harkins - [Event](https://2024julyesipmeeting.sched.com/event/1eVP6) / [Recording](https://youtu.be/T6QAwJIwI3Q?t=3689)
- 2024/05/15 - Pangeo showcase - Tom Nicholas - [Event](https://discourse.pangeo.io/t/pangeo-showcase-virtualizarr-create-virtual-zarr-stores-using-xarray-syntax/4127/2) / [Recording](https://youtu.be/ioxgzhDaYiE) / [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-create-virtual-zarr-stores-using-xarray-syntax)

## Credits

This package was originally developed by [Tom Nicholas](https://github.com/TomNicholas) whilst working at [[C]Worthy](https://cworthy.org), who deserve credit for allowing him to prioritise a generalizable open-source solution to the dataset virtualization problem. VirtualiZarr is now a community-owned multi-stakeholder project.

## Licence

Apache 2.0

## References

[^1]: [_Cloud-Native Repositories for Big Scientific Data_, Abernathey et. al., _Computing in Science & Engineering_.](https://ieeexplore.ieee.org/abstract/document/9354557)

[^2]: (Pronounced like "virtualizer" but more piratey 🦜)
