# Usage

This page explains how to use VirtualiZarr. To understand how the functionality actually works, read the page on [Data Structures](data_structures.md).

## Opening files as virtual datasets

VirtualiZarr is for manipulating "virtual" references to pre-existing data stored on disk in a variety of formats, by representing it in terms of the [Zarr data model](https://zarr-specs.readthedocs.io/en/latest/specs.html) of chunked N-dimensional arrays.

If we have a pre-existing netCDF file on disk:

```python
import xarray as xr

# create an example pre-existing netCDF4 file
ds = xr.tutorial.open_dataset('air_temperature')
ds.to_netcdf('air.nc')
```

We can open a virtual representation of this file using [virtualizarr.open_virtual_dataset][].

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset('air.nc')
```

!!! note
    We did not have to explicitly indicate the file format because [virtualizarr.open_virtual_dataset][] will attempt to automatically infer it.

Printing this "virtual dataset" shows that although it is an instance of `xarray.Dataset`, unlike a typical xarray dataset, in addition to a few in-memory numpy arrays, it also wraps [virtualizarr.manifests.ManifestArray][] objects. You can learn more about the `ManifestArray` class in the [Data Structures documentation](data_structures.md).

```python
vds
```

```
<xarray.Dataset> Size: 31MB
Dimensions:  (lat: 25, lon: 53, time: 2920)
Coordinates:
  * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
  * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
Data variables:
    air      (time, lat, lon) float64 31MB ManifestArray<shape=(2920, 25, 53)...
Attributes:
    Conventions:  COARDS
    title:        4x daily NMC reanalysis (1948)
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
```

Generally a "virtual dataset" is any `xarray.Dataset` which wraps one or more [virtualizarr.manifests.ManifestArray][] objects.

These particular [virtualizarr.manifests.ManifestArray][] objects are each a virtual reference to some data in the `air.nc` netCDF file, with the references stored in the form of "Chunk Manifests".

As the manifest contains only addresses at which to find large binary chunks, the virtual dataset takes up far less space in memory than the original dataset does:

```python
ds.nbytes
```

```
30975672
```

```python
vds.virtualize.nbytes
```

```
23704
```

!!! important

    Virtual datasets are not normal xarray datasets!

    Although the top-level type is still `xarray.Dataset`, they are intended only as an abstract representation of a set of data files, not as something you can do analysis with.
    If you try to load, view, or plot any data you will get a `NotImplementedError`.
    Virtual datasets only support a very limited subset of normal xarray operations, particularly functions and methods for concatenating, merging and extracting variables, as well as operations for renaming dimensions and variables.

    _The only use case for a virtual dataset is [combining references](#combining-virtual-datasets) to files before [writing out those references to disk](#writing-virtual-stores-to-disk)._

### Opening remote files

To open remote files as virtual datasets pass the `reader_options` options, e.g.

```python
aws_credentials = {"key": ..., "secret": ...}
vds = open_virtual_dataset("s3://some-bucket/file.nc", reader_options={'storage_options': aws_credentials})
```

## Loading variables

Once a virtual dataset is created, you won't be able to load the values of the virtual variables into memory.
Instead, you could load specific variables during virtual dataset creation using the regular syntax of `xr.open_dataset`.
Loading the variables during virtual dataset creation has several benefits detailed in the [FAQ](faq.md#why-would-i-want-to-load-variables-using-loadable_variables).

You can use the `loadable_variables` argument to specify variables to load as regular variables rather than virtual variables:

```python
vds = open_virtual_dataset('air.nc', loadable_variables=['air', 'time'])
```

```python
<xarray.Dataset> Size: 31MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
Data variables:
    air      (time, lat, lon) float64 31MB ...
Attributes:
    Conventions:  COARDS
    title:        4x daily NMC reanalysis (1948)
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
```

You can see that the dataset contains a mixture of virtual variables backed by `ManifestArray` objects (`lat` and `lon`), and loadable variables backed by (lazy) numpy arrays (`air` and `time`).

The default value of `loadable_variables` is `None`, which effectively specifies all the "dimension coordinates" in the file, i.e. all one-dimensional coordinate variables whose name is the same as the name of their dimensions. Xarray indexes will also be automatically created for these variables. Together these defaults mean that your virtual dataset will be opened with the same indexes as it would have been if it had been opened with just `xarray.open_dataset()`.

!!! note
    In general, it is recommended to load all of your low-dimensional (e.g scalar and 1D) variables.

    Whilst this does mean the original data will be duplicated in your new virtual zarr store, by loading your coordinates into memory they can be inlined in the reference file, or be stored as single chunks rather than large numbers of extremely tiny chunks. (Both of which will speed up loading that data back when you re-open the virtual store later.)

    However, you should not do this for much higher-dimensional variables, as then you might use a lot of storage duplicating them, defeating the point of the virtual zarr approach.

    Also, anything duplicated could become out of sync with the referenced original files, especially if not using a transactional storage engine such as `Icechunk`.

### Loading CF-encoded time variables

To decode time variables according to the CF conventions upon loading, you must ensure that variable is one of the `loadable_variables` and the `decode_times` argument of `open_virtual_dataset` is set to `True` (`decode_times` defaults to None).

```python
vds = open_virtual_dataset(
    'air.nc',
    loadable_variables=['air', 'time'],
    decode_times=True,
)
```

```python
<xarray.Dataset> Size: 31MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
Data variables:
    air      (time, lat, lon) float64 31MB ...
Attributes:
    Conventions:  COARDS
    title:        4x daily NMC reanalysis (1948)
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
```

## Combining virtual datasets

In general we should be able to combine all the datasets from our archival files into one using some combination of calls to `xarray.concat` and `xarray.merge`.
For combining along multiple dimensions in one call we also have `xarray.combine_nested` and `xarray.combine_by_coords`.
If you're not familiar with any of these functions we recommend you skim through [xarray's docs on combining](https://docs.xarray.dev/en/stable/user-guide/combining.html).

Let's create two new netCDF files, which we would need to open and concatenate in a specific order to represent our entire dataset.

```python
ds1 = ds.isel(time=slice(None, 1460))
ds2 = ds.isel(time=slice(1460, None))

ds1.to_netcdf('air1.nc')
ds2.to_netcdf('air2.nc')
```

Note that we have created these in such a way that each dataset has one equally-sized chunk.

!!! important
    Currently the virtual approach requires the same chunking and encoding across datasets. See the [FAQ](faq.md#can-my-specific-data-be-virtualized) for more details.

### Manual concatenation ordering

The simplest case of concatenation is when you have a set of files and you know the order in which they should be concatenated, _without looking inside the files_.
In this case it is sufficient to open the files one-by-one, then pass the virtual datasets as a list to the concatenation function.

```python
vds1 = open_virtual_dataset('air1.nc')
vds2 = open_virtual_dataset('air2.nc')
```

As we know the correct order a priori, we can just combine along one dimension using `xarray.concat`.

```python
combined_vds = xr.concat([vds1, vds2], dim='time')
combined_vds
```

```
<xarray.Dataset> Size: 31MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
  * lat      (lat) float32 100B 75.0 72.5 70.0 67.5 65.0 ... 22.5 20.0 17.5 15.0
  * lon      (lon) float32 212B 200.0 202.5 205.0 207.5 ... 325.0 327.5 330.0
  * time     (time) datetime64[ns] 23kB 2013-01-01 ... 2014-12-31T18:00:00
Data variables:
    air      (time, lat, lon) float64 31MB ManifestArray<shape=(2920, 25, 53)...
Attributes:
    Conventions:  COARDS
    title:        4x daily NMC reanalysis (1948)
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
```

We can see that the resulting combined manifest has two chunks, as expected.

```python
combined_vds['air'].data.manifest.dict()
```

```
{'0.0.0': {'path': 'file:///work/data/air1.nc', 'offset': 15419, 'length': 3869000},
 '1.0.0': {'path': 'file:///work/data/air2.nc', 'offset': 15419, 'length': 3869000}}
```

!!! note
    If you have any virtual coordinate variables, you will likely need to specify the keyword arguments `coords='minimal'` and `compat='override'` to `xarray.concat()`, because the default behaviour of xarray will attempt to load coordinates in order to check their compatibility with one another.

    In future this [default will be changed](https://github.com/pydata/xarray/issues/8778), such that passing these two arguments explicitly will become unnecessary.

The general multi-dimensional version of this concatenation-by-order-supplied can be achieved using `xarray.combine_nested()`.

```python
combined_vds = xr.combine_nested([vds1, vds2], concat_dim=['time'])
```

In N-dimensions the datasets would need to be passed as an N-deep nested list-of-lists, see the [xarray docs](https://docs.xarray.dev/en/stable/user-guide/combining.html#combining-along-multiple-dimensions).

!!! note
    For manual concatenation we can actually avoid creating any xarray indexes, as we won't need them.
    Without indexes we can avoid loading any data whatsoever from the files.
    However, you should first be confident that the archival files actually do have compatible data, as the coordinate values then cannot be efficiently compared for consistency (i.e. aligned).

You can achieve both the opening and combining steps for multiple files in one go by using [open_virtual_mfdataset][virtualizarr.open_virtual_mfdataset].

```python
combined_vds = xr.open_virtual_mfdataset(['air1.nc', 'air2.nc'], concat_dim='time', combine='nested')
```

We passed `combine='nested'` to specify that we want the datasets to be combined in the order they appear, using `xr.combine_nested` under the hood.

### Ordering by coordinate values

If you're happy to load 1D dimension coordinates into memory, you can use their values to do the ordering for you!

```python
vds1 = open_virtual_dataset('air1.nc')
vds2 = open_virtual_dataset('air2.nc')

combined_vds = xr.combine_by_coords([vds2, vds1])
```

Notice we don't have to specify the concatenation dimension explicitly - xarray works out the correct ordering for us.
Even though we actually passed in the virtual datasets in the wrong order just now, they have been combined in the correct order such that the 1-dimensional `time` coordinate has ascending values.
As a result our chunk manifest still has the chunks listed in the expected order:

```python
combined_vds['air'].data.manifest.dict()
```

```
{'0.0.0': {'path': 'file:///work/data/air1.nc', 'offset': 15419, 'length': 3869000},
 '1.0.0': {'path': 'file:///work/data/air2.nc', 'offset': 15419, 'length': 3869000}}
```

Again, we can achieve both the opening and combining steps for multiple files in one go by using [open_virtual_mfdataset][virtualizarr.open_virtual_mfdataset], but this passing `combine='by_coords'`.

```python
combined_vds = xr.open_virtual_mfdataset(['air2.nc', 'air1.nc'], combine='by_coords')
```

We can even pass in a glob to find all the files we want to automatically combine:

```python
combined_vds = xr.open_virtual_mfdataset('air*.nc', combine='by_coords')
```

### Ordering using metadata

TODO: Use preprocess to create a new index from the metadata. Requires `open_virtual_mfdataset` to be implemented in [PR #349](https://github.com/zarr-developers/VirtualiZarr/pull/349).

### Combining many virtual datasets at once

Combining a large number (e.g., 1000s) of virtual datasets at once should be very quick (a few seconds), as we are manipulating only a few KBs of metadata in memory.

However creating 1000s of virtual datasets at once can take a very long time.
(If it were quick to do so, there would be little need for this library!)
See the page on [Scaling](scaling.md) for tips on how to create large numbers of virtual datasets at once.

## Writing virtual stores to disk

Once we've combined references to all the chunks of all our archival files into one virtual xarray dataset, we still need to write these references out to disk so that they can be read by our analysis code later.

### Writing to Kerchunk's format and reading data via fsspec

The [kerchunk library](https://github.com/fsspec/kerchunk) has its own [specification](https://fsspec.github.io/kerchunk/spec.html) for how byte range references should be serialized (either as a JSON or parquet file).

To write out all the references in the virtual dataset as a single kerchunk-compliant JSON or parquet file, you can use the [virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk][] accessor method.

```python
combined_vds.virtualize.to_kerchunk('combined.json', format='json')
```

These zarr-like references can now be interpreted by [fsspec](https://github.com/fsspec/filesystem_spec), using kerchunk's built-in xarray backend (kerchunk must be installed to use `engine='kerchunk'`).

```python
combined_ds = xr.open_dataset('combined.json', engine="kerchunk")
```

In-memory ("loadable") variables backed by numpy arrays can also be written out to kerchunk reference files, with the values serialized as bytes.
This is equivalent to kerchunk's concept of "inlining", but done on a per-array basis using the `loadable_variables` kwarg rather than a per-chunk basis using kerchunk's `inline_threshold` kwarg.

!!! note
    Currently you can only serialize in-memory variables to kerchunk references if they do not have any encoding.

When you have many chunks, the reference file can get large enough to be unwieldy as JSON.
In that case the references can be instead stored as parquet, which again this uses kerchunk internally.

```python
combined_vds.virtualize.to_kerchunk('combined.parquet', format='parquet')
```

And again we can read these references using the "kerchunk" backend

```python
combined_ds = xr.open_dataset('combined.parquet', engine="kerchunk")
```

By default references are placed in separate parquet files when the total number of references exceeds `record_size`.
If there are fewer than `categorical_threshold` unique urls referenced by a particular variable, url will be stored as a categorical variable.

### Writing to an Icechunk Store

We can also write these references out as an [IcechunkStore](https://icechunk.io/).
`Icechunk` is an open-source, cloud-native transactional tensor storage engine that is fully compatible with Zarr-Python version 3, as it conforms to the Zarr V3 specification.
To export our virtual dataset to an `Icechunk` Store, we simply use the [virtualizarr.VirtualiZarrDatasetAccessor.to_icechunk][] accessor method.

```python
# create an icechunk repository, session and write the virtual dataset to the session
import icechunk
storage = icechunk.local_filesystem_storage("./local/icechunk/store")

# By default, local virtual references and public remote virtual references can be read without extra configuration.
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")

# write the virtual dataset to the session with the IcechunkStore
vds1.virtualize.to_icechunk(session.store)
session.commit("Wrote first dataset")
```

#### Append to an existing Icechunk Store

You can append a virtual dataset to an existing Icechunk store using the `append_dim` argument.
This is especially useful for datasets that grow over time.

!!! important
    Note again that the virtual Zarr approach requires the same chunking and encoding across datasets. This including when appending to an existing Icechunk-backed Zarr store. See the [FAQ](faq.md#can-my-specific-data-be-virtualized) for more details.

```python
session = repo.writeable_session("main")

# write the virtual dataset to the session with the IcechunkStore
vds2.virtualize.to_icechunk(session.store, append_dim="time")
session.commit("Appended second dataset")
```

See the [Icechunk documentation](https://icechunk.io/icechunk-python/virtual/#creating-a-virtual-dataset-with-virtualizarr) for more details.

## Opening Kerchunk references as virtual datasets

You can open existing Kerchunk `json` or `parquet` references as Virtualizarr virtual datasets.
This may be useful for manipulating them or converting existing kerchunk-formatted references to other reference storage formats such as [Icechunk](https://icechunk.io/).

```python
vds = open_virtual_dataset('combined.json', filetype='kerchunk')
# or
vds = open_virtual_dataset('combined.parquet', filetype='kerchunk')
```

One difference between the kerchunk references format and virtualizarr's internal manifest representation (as well as Icechunk's format) is that paths in kerchunk references can be relative paths.
Opening kerchunk references that contain relative local filepaths therefore requires supplying another piece of information: the directory of the `fsspec` filesystem which the filepath was defined relative to.

You can dis-ambuiguate kerchunk references containing relative paths by passing the `fs_root` kwarg to `virtual_backend_kwargs`.

```python
# file `relative_refs.json` contains a path like './file.nc'

vds = open_virtual_dataset(
    'relative_refs.json',
    filetype='kerchunk',
    virtual_backend_kwargs={'fs_root': 'file:///some_directory/'}
)

# the path in the virtual dataset will now be 'file:///some_directory/file.nc'
```

Note that as the virtualizarr [virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk][] method only writes absolute paths, the only scenario in which you might come across references containing relative paths is if you are opening references that were previously created using the `kerchunk` library alone.

## Rewriting existing manifests

Sometimes it can be useful to rewrite the contents of an already-generated in-memory manifest or virtual dataset.

### Rewriting file paths

You can rewrite the file paths stored in a manifest or virtual dataset without changing the byte range information using the [virtualizarr.VirtualiZarrDatasetAccessor.rename_paths][] accessor method.

For example, you may want to rename file paths according to a function to reflect having moved the location of the referenced files from local storage to an S3 bucket.

```python
def local_to_s3_url(old_local_path: str) -> str:
    from pathlib import Path

    new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"

    filename = Path(old_local_path).name
    return str(new_s3_bucket_url / filename)

renamed_vds = vds.virtualize.rename_paths(local_to_s3_url)
renamed_vds['air'].data.manifest.dict()
```

```
{'0.0.0': {'path': 'http://s3.amazonaws.com/my_bucket/air.nc', 'offset': 15419, 'length': 7738000}}
```

!!! note
    Icechunk does not yet support [chunk references with HTTP URLs](https://github.com/earth-mover/icechunk/issues/526) (only local files and S3 URIs), so the ability to rename paths can be useful for renaming paths into a form Icechunk accepts.
