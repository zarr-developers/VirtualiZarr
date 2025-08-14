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

## Quick usage example

Creating the virtual dataset looks quite similar to how we normally open data with [xarray][], but there are a few notable differences that are shown through this example.

First, import the necessary functions and classes:

```python exec="on" source="above" session="homepage"
import icechunk
import obstore

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
```

Zarr can emit a lot of warnings about Numcodecs not being including in the Zarr version 3
specification yet -- let's suppress those.

```python exec="on" source="above" session="homepage"
import warnings
warnings.filterwarnings(
  "ignore",
  message="Numcodecs codecs are not in the Zarr version 3 specification*",
  category=UserWarning
)
```

```python exec="on" session="homepage"
# This code isn't shown since we didn't set source="above"
import xarray as xr
xr.set_options(display_style="html")
```

We can use Obstore's [`obstore.store.from_url`][obstore.store.from_url] convenience method to create an [ObjectStore][obstore.store.ObjectStore] that can fetch data from the specified URLs.

```python exec="on" source="above" session="homepage"
bucket = "s3://nex-gddp-cmip6"
path = "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
```

We also need to create an [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] that
maps the URL structure to the ObjectStore.

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
urls = [f"s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_{year}_v2.0.nc" for year in range(2015, 2017)]
vds = open_virtual_mfdataset(urls, parser = parser, registry = registry)
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

See the [Usage docs page](usage.md) for more details.

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
