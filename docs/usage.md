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

We can open a virtual representation of this file using {py:func}`open_virtual_dataset <virtualizarr.open_virtual_dataset>`.

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset('air.nc')
```

(Notice we did not have to explicitly indicate the file format, as {py:func}`open_virtual_dataset <virtualizarr.open_virtual_dataset>` will attempt to automatically infer it.)


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


Generally a "virtual dataset" is any `xarray.Dataset` which wraps one or more {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects.

These particular {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects are each a virtual reference to some data in the `air.nc` netCDF file, with the references stored in the form of "Chunk Manifests".

```{important} Virtual datasets are not normal xarray datasets!

Although the top-level type is still `xarray.Dataset`, they are intended only as an abstract representation of a set of data files, not as something you can do analysis with. If you try to load, view, or plot any data you will get a `NotImplementedError`. Virtual datasets only support a very limited subset of normal xarray operations, particularly functions and methods for concatenating, merging and extracting variables, as well as operations for renaming dimensions and variables.

_The only use case for a virtual dataset is [combining references](#combining-virtual-datasets) to files before [writing out those references to disk](#writing-virtual-stores-to-disk)._
```

### Opening remote files

To open remote files as virtual datasets pass the `reader_options` options, e.g.

```python
aws_credentials = {"key": ..., "secret": ...}
vds = open_virtual_dataset("s3://some-bucket/file.nc", reader_options={'storage_options': aws_credentials})
```

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
{'0.0.0': {'path': 'file:///work/data/air.nc', 'offset': 15419, 'length': 7738000}}
```

In this case we can see that the `"air"` variable contains only one chunk, the bytes for which live in the `file:///work/data/air.nc` file, at the location given by the `'offset'` and `'length'` attributes.

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
{'0.0.0': {'path': 'file:///work/data/air.nc', 'offset': 15419, 'length': 7738000},
 '1.0.0': {'path': 'file:///work/data/air.nc', 'offset': 15419, 'length': 7738000}}
```

This concatenation property is what will allow us to combine the data from multiple netCDF files on disk into a single Zarr store containing arrays of many chunks.

```{note}
As a single Zarr array has only one array-level set of compression codecs by definition, concatenation of arrays from files saved to disk with differing codecs cannot be achieved through concatenation of `ManifestArray` objects. Implementing this feature will require a more abstract and general notion of concatenation, see [GH issue #5](https://github.com/TomNicholas/VirtualiZarr/issues/5).
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

The full Zarr model (for a single group) includes multiple arrays, array names, named dimensions, and arbitrary dictionary-like attrs on each array. Whilst the duck-typed `ManifestArray` cannot store all of this information, an `xarray.Dataset` wrapping multiple `ManifestArray`s maps neatly to the Zarr model. This is what the virtual dataset we opened represents - all the information in one entire Zarr group, but held as references to on-disk chunks instead of as in-memory arrays.

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
{'0.0.0': {'path': 'file:///work/data/air1.nc', 'offset': 15419, 'length': 3869000},
 '1.0.0': {'path': 'file:///work/data/air2.nc', 'offset': 15419, 'length': 3869000}}
```

```{note}
The keyword arguments `coords='minimal', compat='override'` are currently necessary because the default behaviour of xarray will attempt to load coordinates in order to check their compatibility with one another. In future this [default will be changed](https://github.com/pydata/xarray/issues/8778), such that passing these two arguments explicitly will become unnecessary.
```

The general multi-dimensional version of this concatenation-by-order-supplied can be achieved using `xarray.combine_nested`.

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

TODO: Reinstate this part of the docs once [GH issue #18](https://github.com/TomNicholas/VirtualiZarr/issues/18#issuecomment-2023955860) is properly closed.

### Automatic ordering using metadata

TODO: Use preprocess to create a new index from the metadata

## Loading variables

Whilst the values of virtual variables (i.e. those backed by `ManifestArray` objects) cannot be loaded into memory, you do have the option of opening specific variables from the file as loadable lazy numpy/dask arrays, just like `xr.open_dataset` normally returns. These variables are specified using the `loadable_variables` argument:

```python
vds = open_virtual_dataset('air.nc', loadable_variables=['air', 'time'], indexes={})
```
```python
<xarray.Dataset> Size: 31MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
  Data variables:
    air      (time, lat, lon) float64 31MB ...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```
You can see that the dataset contains a mixture of virtual variables backed by `ManifestArray` objects (`lat` and `lon`), and loadable variables backed by (lazy) numpy arrays (`air` and `time`).

Loading variables can be useful in a few scenarios:
1. You need to look at the actual values of a multi-dimensional variable in order to decide what to do next,
2. Storing a variable on-disk as a set of references would be inefficient, e.g. because it's a very small array (saving the values like this is similar to kerchunk's concept of "inlining" data),
3. The variable has encoding, and the simplest way to decode it correctly is to let xarray's standard decoding machinery load it into memory and apply the decoding.

### CF-encoded time variables

To correctly decode time variables according to the CF conventions, you need to pass `time` to `loadable_variables` and ensure the `decode_times` argument of `open_virtual_dataset` is set to True (`decode_times` defaults to None).

```python
vds = open_virtual_dataset(
    'air.nc',
    loadable_variables=['air', 'time'],
    decode_times=True,
    indexes={},
)
```
```python
<xarray.Dataset> Size: 31MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
    time     (time) datetime64[ns] 23kB 2013-01-01T00:02:06.757437440 ... 201...
Data variables:
    air      (time, lat, lon) float64 31MB ...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```

## Writing virtual stores to disk

Once we've combined references to all the chunks of all our legacy files into one virtual xarray dataset, we still need to write these references out to disk so that they can be read by our analysis code later.

### Writing to Kerchunk's format and reading data via fsspec

The [kerchunk library](https://github.com/fsspec/kerchunk) has its own [specification](https://fsspec.github.io/kerchunk/spec.html) for how byte range references should be serialized (either as a JSON or parquet file).

To write out all the references in the virtual dataset as a single kerchunk-compliant JSON or parquet file, you can use the {py:meth}`vds.virtualize.to_kerchunk <virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk>` accessor method.

```python
combined_vds.virtualize.to_kerchunk('combined.json', format='json')
```

These references can now be interpreted like they were a Zarr store by [fsspec](https://github.com/fsspec/filesystem_spec), using kerchunk's built-in xarray backend (kerchunk must be installed to use `engine='kerchunk'`).

```python
combined_ds = xr.open_dataset('combined.json', engine="kerchunk")
```

In-memory ("loadable") variables backed by numpy arrays can also be written out to kerchunk reference files, with the values serialized as bytes. This is equivalent to kerchunk's concept of "inlining", but done on a per-array basis using the `loadable_variables` kwarg rather than a per-chunk basis using kerchunk's `inline_threshold` kwarg.

```{note}
Currently you can only serialize in-memory variables to kerchunk references if they do not have any encoding.
```

When you have many chunks, the reference file can get large enough to be unwieldy as json. In that case the references can be instead stored as parquet. Again this uses kerchunk internally.

```python
combined_vds.virtualize.to_kerchunk('combined.parquet', format='parquet')
```

And again we can read these references using the "kerchunk" backend as if it were a regular Zarr store

```python
combined_ds = xr.open_dataset('combined.parquet', engine="kerchunk")
```

By default references are placed in separate parquet file when the total number of references exceeds `record_size`. If there are fewer than `categorical_threshold` unique urls referenced by a particular variable, url will be stored as a categorical variable.

### Writing to an Icechunk Store

We can also write these references out as an [IcechunkStore](https://icechunk.io/). `Icechunk` is a Open-source, cloud-native transactional tensor storage engine that is compatible with zarr version 3. To export our virtual dataset to an `Icechunk` Store, we simply use the {py:meth}`vds.virtualize.to_icechunk <virtualizarr.VirtualiZarrDatasetAccessor.to_icechunk>` accessor method.

```python
# create an icechunk store
from icechunk import IcechunkStore, StorageConfig, StoreConfig, VirtualRefConfig
storage = StorageConfig.filesystem(str('combined'))
store = IcechunkStore.create(storage=storage, mode="w", config=StoreConfig(
    virtual_ref_config=VirtualRefConfig.s3_anonymous(region='us-east-1'),
))

combined_vds.virtualize.to_icechunk(store)
```

See the [Icechunk documentation](https://icechunk.io/icechunk-python/virtual/#creating-a-virtual-dataset-with-virtualizarr) for more details.

### Writing as Zarr

Alternatively, we can write these references out as an actual Zarr store, at least one that is compliant with the [proposed "Chunk Manifest" ZEP](https://github.com/zarr-developers/zarr-specs/issues/287). To do this we simply use the {py:meth}`vds.virtualize.to_zarr <virtualizarr.VirtualiZarrDatasetAccessor.to_zarr>` accessor method.

```python
combined_vds.virtualize.to_zarr('combined.zarr')
```

The result is a zarr v3 store on disk which contains the chunk manifest information written out as `manifest.json` files, so the store looks like this:

```
combined/zarr.json  <- group metadata
combined/air/zarr.json  <- array metadata
combined/air/manifest.json <- array manifest
...
```

The advantage of this format is that any zarr v3 reader that understands the chunk manifest ZEP could read from this store, no matter what language it is written in (e.g. via `zarr-python`, `zarr-js`, or rust). This reading would also not require `fsspec`.

```{note}
Currently there are not yet any zarr v3 readers which understand the chunk manifest ZEP, so until then this feature cannot be used for data processing.

This store can however be read by {py:func}`~virtualizarr.open_virtual_dataset`, by passing `filetype="zarr_v3"`.
```

## Opening Kerchunk references as virtual datasets

You can open existing Kerchunk `json` or `parquet` references as Virtualizarr virtual datasets. This may be useful for converting existing Kerchunk formatted references to storage formats like [Icechunk](https://icechunk.io/).

```python
vds = open_virtual_dataset('combined.json', filetype='kerchunk', indexes={})
# or
vds = open_virtual_dataset('combined.parquet', filetype='kerchunk', indexes={})
```

One difference between the kerchunk references format and virtualizarr's internal manifest representation (as well as icechunk's format) is that paths in kerchunk references can be relative paths. Opening kerchunk references that contain relative local filepaths therefore requires supplying another piece of information: the directory of the ``fsspec`` filesystem which the filepath was defined relative to.

You can dis-ambuiguate kerchunk references containing relative paths by passing the ``fs_root`` kwarg to ``virtual_backend_kwargs``.

```python
# file `relative_refs.json` contains a path like './file.nc'

vds = open_virtual_dataset(
    'relative_refs.json',
    filetype='kerchunk',
    indexes={},
    virtual_backend_kwargs={'fs_root': 'file:///some_directory/'}
)

# the path in the virtual dataset will now be 'file:///some_directory/file.nc'
```

Note that as the virtualizarr {py:meth}`vds.virtualize.to_kerchunk <virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk>` method only writes absolute paths, the only scenario in which you might come across references containing relative paths is if you are opening references that were previously created using the ``kerchunk`` library alone.

## Rewriting existing manifests

Sometimes it can be useful to rewrite the contents of an already-generated manifest or virtual dataset.

### Rewriting file paths

You can rewrite the file paths stored in a manifest or virtual dataset without changing the byte range information using the {py:meth}`vds.virtualize.rename_paths <virtualizarr.VirtualiZarrDatasetAccessor.rename_paths>` accessor method.

For example, you may want to rename file paths according to a function to reflect having moved the location of the referenced files from local storage to an S3 bucket.

```python
def local_to_s3_url(old_local_path: str) -> str:
    from pathlib import Path

    new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"

    filename = Path(old_local_path).name
    return str(new_s3_bucket_url / filename)
```
```python
renamed_vds = vds.virtualize.rename_paths(local_to_s3_url)
renamed_vds['air'].data.manifest.dict()
```
```
{'0.0.0': {'path': 'http://s3.amazonaws.com/my_bucket/air.nc', 'offset': 15419, 'length': 7738000}}
```
