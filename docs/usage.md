(usage)=
# Usage

This page explains how to use VirtualiZarr today, by introducing the key concepts one-by-one.

## Opening files as virtual datasets

VirtualiZarr is for manipulating "virtual" references to pre-existing data stored on disk in a variety of formats, by representing it in terms of the [Zarr data model](https://zarr-specs.readthedocs.io/en/latest/specs.html) of chunked N-dimensional arrays.

If we have a pre-existing netCDF file on disk,

```python
import xarray as xr

# create an example pre-existing netCDF4 file
ds = xr.tutorial.open_dataset('air_temperature')
ds.to_netcdf('air.nc')
```

We can open a virtual representation of this file using {py:func}`open_virtual_dataset <virtualizarr.xarray.open_virtual_dataset>`.

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset('air.nc')
```

(Notice we did not have to explicitly indicate the file format, as {py:func}`open_virtual_dataset <virtualizarr.xarray.open_virtual_dataset>` will attempt to automatically infer it.)

```{note}
In future we would like for it to be possible to just use `xr.open_dataset`, e.g.

    import virtualizarr

    vds = xr.open_dataset('air.nc', engine='virtualizarr')

but this requires some [upstream changes](https://github.com/TomNicholas/VirtualiZarr/issues/35) in xarray.
```

Printing this "virtual dataset" shows that although it is an instance of `xarray.Dataset`, unlike a typical xarray dataset, it does not contain numpy or dask arrays, but instead it wraps {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects.

```python
vds
```
```
<xarray.Dataset> Size: 8MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
    time     (time) float32 12kB ManifestArray<shape=(2920,), dtype=float32, ...
Data variables:
    air      (time, lat, lon) int16 8MB ManifestArray<shape=(2920, 25, 53), d...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```

These {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects are each a virtual reference to some data in the `air.nc` netCDF file, with the references stored in the form of "Chunk Manifests".

## Chunk Manifests

In the Zarr model N-dimensional arrays are stored as a series of compressed chunks, each labelled by a chunk key which indicates its position in the array. Whilst conventionally each of these Zarr chunks are a separate compressed binary file stored within a Zarr Store, there is no reason why these chunks could not actually already exist as part of another file (e.g. a netCDF file), and be loaded by reading a specific byte range from this pre-existing file.

A "Chunk Manifest" is a list of chunk keys and their corresponding byte ranges in specific files, grouped together such that all the chunks form part of one Zarr-like array. For example, a chunk manifest for a 3-dimensional array made up of 4 chunks might look like this:

```python
{
    "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
    "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
    "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
    "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
}
```

Notice that the `"path"` attribute points to a netCDF file `"foo.nc"` stored in a remote S3 bucket. There is no need for the files the chunk manifest refers to to be local.

Our virtual dataset we opened above contains multiple chunk manifests stored in-memory, which we can see by pulling one out as a python dictionary.

```python
marr = vds['air'].data
manifest = marr.manifest
manifest.dict()
```
```python
{'0.0.0': {'path': 'air.nc', 'offset': 15419, 'length': 7738000}}
```

In this case we can see that the `"air"` variable contains only one chunk, the bytes for which live in the `air.nc` file at the location given by the `'offset'` and `'length'` attributes.

The {py:class}`ChunkManifest <virtualizarr.manifests.ChunkManifest>` class is virtualizarr's internal in-memory representation of this manifest.

## `ManifestArray` class

A Zarr array is defined not just by the location of its constituent chunk data, but by its array-level attributes such as `shape` and `dtype`. The {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` class stores both the array-level attributes and the corresponding chunk manifest.

```python
marr
```
```
ManifestArray<shape=(2920, 25, 53), dtype=int16, chunks=(2920, 25, 53)>
```
```python
marr.manifest
```
```
ChunkManifest<shape=(1, 1, 1)>
```
```python
marr.zarray
```
```
ZArray(shape=(2920, 25, 53), chunks=(2920, 25, 53), dtype=int16, compressor=None, filters=None, fill_value=None)
```

A `ManifestArray` can therefore be thought of as a virtualized representation of a single Zarr array.

As it defines various array-like methods, a `ManifestArray` can often be treated like a ["duck array"](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html). In particular, concatenation of multiple `ManifestArray` objects can be done via merging their chunk manifests into one (and re-labelling the chunk keys).

```python
import numpy as np

concatenated = np.concatenate([marr, marr], axis=0)
concatenated
```
```
ManifestArray<shape=(5840, 25, 53), dtype=int16, chunks=(2920, 25, 53)>
```
```python
concatenated.manifest.dict()
```
```
{'0.0.0': {'path': 'air.nc', 'offset': 15419, 'length': 7738000},
 '1.0.0': {'path': 'air.nc', 'offset': 15419, 'length': 7738000}}
```

This concatenation property is what will allow us to combine the data from multiple netCDF files on disk into a single Zarr store containing arrays of many chunks.

```{note}
As a single Zarr array has only one array-level set of compression codecs by definition, concatenation of arrays from files saved to disk with differing codecs cannot be achieved through concatenation of `ManifestArray` objects. Implementing this feature will require a more abstract and general notion of concatentation, see [GH issue #5](https://github.com/TomNicholas/VirtualiZarr/issues/5).
```

Remember that you cannot load values from a `ManifestArray` directly.

```python
vds['air'].values
```
```python
NotImplementedError: ManifestArrays can't be converted into numpy arrays or pandas Index objects
```

The whole point is to manipulate references to the data without actually loading any data.

```{note}
You also cannot currently index into a `ManifestArray`, as arbitrary indexing would require loading data values to create the new array. We could imagine supporting indexing without loading data when slicing only along chunk boundaries, but this has not yet been implemented (see [GH issue #51](https://github.com/TomNicholas/VirtualiZarr/issues/51)).
```

## Virtual Datasets as Zarr Groups

The full Zarr model (for a single group) includes multiple arrays, array names, named dimensions, and arbitrary dictionary-like attrs on each array. Whilst the duck-typed `ManifestArray` cannot store all of this information, an `xarray.Dataset` wrapping multiple `ManifestArray`s maps really nicely to the Zarr model. This is what the virtual dataset we opened represents - all the information in one entire Zarr group, but held as references to on-disk chunks instead of in-memory arrays.

The problem of combining many legacy format files (e.g. netCDF files) into one virtual Zarr store therefore becomes just a matter of opening each file using `open_virtual_dataset` and using [xarray's various combining functions](https://docs.xarray.dev/en/stable/user-guide/combining.html) to combine them into one aggregate virtual dataset.

## Combining virtual datasets

In general we should be able to combine all the datasets from our legacy files into one using some combination of calls to `xarray.concat` and `xarray.merge`. For combining along multiple dimensions in one call we also have `xarray.combine_nested` and `xarray.combine_by_coords`. If you're not familiar with any of these functions we recommend you skim through [xarray's docs on combining](https://docs.xarray.dev/en/stable/user-guide/combining.html).

Let's create two new netCDF files, which we would need to open and concatenate in a specific order to represent our entire dataset.

```python
ds1 = ds.isel(time=slice(None, 1460))
ds2 = ds.isel(time=slice(1460, None))

ds1.to_netcdf('air1.nc')
ds2.to_netcdf('air2.nc')
```

Note that we have created these in such a way that each dataset has one equally-sized chunk.

TODO: Note about variable-length chunking?

### Manual concatenation ordering

The simplest case of concatenation is when you have a set of files and you know which order they should be concatenated in, _without looking inside the files_. In this case it is sufficient to open the files one-by-one, then pass the virtual datasets as a list to the concatenation function.

We can actually avoid creating any xarray indexes, as we won't need them. Without indexes we can avoid loading any data whatsoever from the files, making our opening and combining much faster than it normally would be. **Therefore if you can do your combining manually you should.** However, you should first be confident that the legacy files actually do have compatible data, as only the array shapes and dimension names will be checked for consistency.

You can specify that you don't want any indexes to be created by passing `indexes={}` to `open_virtual_dataset`.

```python
vds1 = open_virtual_dataset('air1.nc', indexes={})
vds2 = open_virtual_dataset('air2.nc', indexes={})
```

We can see that the datasets have no indexes.

```python
vds1.indexes
```
```
Indexes:
    *empty*
```

```{note}
Passing `indexes={}` will only work if you use a [specific branch of xarray](https://github.com/TomNicholas/xarray/tree/concat-no-indexes), as it requires multiple in-progress PR's, see [GH issue #14](https://github.com/TomNicholas/VirtualiZarr/issues/14#issuecomment-2018369470).
```

As we know the correct order a priori, we can just combine along one dimension using `xarray.concat`.

```
combined_vds = xr.concat([vds1, vds2], dim='time', coords='minimal', compat='override')
combined_vds
```
```
<xarray.Dataset> Size: 8MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
    time     (time) float32 12kB ManifestArray<shape=(2920,), dtype=float32, ...
Data variables:
    air      (time, lat, lon) int16 8MB ManifestArray<shape=(2920, 25, 53), d...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```

We can see that the resulting combined manifest has two chunks, as expected.

```python
combined_vds['air'].data.manifest.dict()
```
```
{'0.0.0': {'path': 'air1.nc', 'offset': 15419, 'length': 3869000},
 '1.0.0': {'path': 'air2.nc', 'offset': 15419, 'length': 3869000}}
```

```{note}
The keyword arguments `coords='minimal', compat='override'` are currently necessary because the default behaviour of xarray will attempt to load coordinates in order to check their compatibility with one another. In future this [default will be changed](https://github.com/pydata/xarray/issues/8778), such that passing these two arguments explicitly will become unnecessary.
```

The general multi-dimensional version of this contatenation-by-order-supplied can be achieved using `xarray.combine_nested`.

```python
combined_vds = xr.combine_nested([vds1, vds2], concat_dim=['time'], coords='minimal', compat='override')
```

In N-dimensions the datasets would need to be passed as an N-deep nested list-of-lists, see the [xarray docs](https://docs.xarray.dev/en/stable/user-guide/combining.html#combining-along-multiple-dimensions).

```{note}
In future we would like for it to be possible to just use `xr.open_mfdataset` to open the files and combine them in one go, e.g.

    vds = xr.open_mfdataset(
        ['air1.nc', 'air2.nc'],
        combine='nested',
        concat_dim=['time'],
        coords='minimal',
        compat='override',
        indexes={},
    )

but this requires some [upstream changes](https://github.com/TomNicholas/VirtualiZarr/issues/35) in xarray.
```

### Automatic ordering using coordinate data

Sometimes we don't have a priori knowledge of which files contain what content, and we would like to concatenate them in an order dictated by their coordinates (e.g. so that a `time` coordinate monotonically increases into the future).

For this we will actually want to create xarray indexes, so that we can use the values in them to determine the correct concatenation order. This requires loading coordinate values into memory, the same way that `xarray.open_dataset` does by default.

To open a virtual dataset but with in-memory indexes along 1D [dimension coordinates](), pass `indexes=None` to `open_virtual_dataset` (which is the default).

```python
vds1 = open_virtual_dataset('air1.nc')
vds2 = open_virtual_dataset('air2.nc')
```

Now we can see that some indexes have been created by default.

```python
vds1.xindexes
```
```
Indexes:
    lat      PandasIndex
    lon      PandasIndex
    time     PandasIndex
```

To use these indexes to infer concatenation order we can use `xarray.combine_by_coords`.

```python
combined_vds = xr.combine_by_coords([vds2, vds1])
combined_vds
```
```
<xarray.Dataset> Size: 8MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
  * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
  * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
Data variables:
    air      (time, lat, lon) int16 8MB ManifestArray<shape=(2920, 25, 53), d...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```
We can see that despite the fact we passed the datasets out of order, the time coordinate in the result is still ordered correctly.

Note that we can safely omit the `compat='override'` kwarg now, because we have indexes whose values will be compared.

TODO: Improve xarray's error message for if we tried to use `combine_by_coords` without creating indexes first.

```{note}
In future we would like for it to be possible to just use `xr.open_mfdataset` to open the files and combine them in one go, e.g.

    vds = xr.open_mfdataset(
        ['air2.nc', 'air1.nc'],
        combine='by_coords',
    )

but this requires some [upstream changes](https://github.com/TomNicholas/VirtualiZarr/issues/35) in xarray.
```

### Automatic ordering using metadata

TODO: Use preprocess to create a new index from the metadata

## Writing virtual stores to disk

Once we've combined references to all the chunks of all our legacy files into one virtual xarray dataset, we still need to write these references out to disk so that they can be read by our analysis code later.

### Writing to Kerchunk's format and reading via fsspec

The [kerchunk library](https://github.com/fsspec/kerchunk) has its own [specification](https://fsspec.github.io/kerchunk/spec.html) for how byte range references should be serialized (either as a JSON or parquet file).

To write out all the references in the virtual dataset as a single kerchunk-compliant JSON file, you can use the {py:meth}`ds.virtualize.to_kerchunk <virtualizarr.xarray.VirtualiZarrDatasetAccessor.to_kerchunk>` accessor method.

```python
combined_vds.virtualize.to_kerchunk('combined.json', format='json')
```

These references can now be interpreted like they were a Zarr store by [fsspec](https://github.com/fsspec/filesystem_spec), using its built-in kerchunk xarray backend.

```python
import fsspec

fs = fsspec.filesystem("reference", fo=f"combined.json")
mapper = fs.get_mapper("")

combined_ds = xr.open_dataset(mapper, engine="kerchunk")
```

### Writing as Zarr

TODO: Explanation of how this requires changes in zarr upstream to be able to read it
