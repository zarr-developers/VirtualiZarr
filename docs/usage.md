# Usage

This page explains how to use VirtualiZarr. Review the [Data Structures](data_structures.md) documentation if you want to understand the conceptual models underpinning VirtualiZarr.

## Opening files as virtual datasets

VirtualiZarr is for creating and manipulating "virtual" references to pre-existing data stored in the cloud or on disk in a variety of formats, by representing it in terms of the [Zarr data model](https://zarr-specs.readthedocs.io/en/latest/specs.html) of chunked N-dimensional arrays.

The first step to virtualizing data is to create an [ObjectStore][obstore.store.ObjectStore] instance
that can access your data. Available ObjectStores are described in the [obstore docs](https://developmentseed.org/obstore/latest/getting-started/#constructing-a-store).


!!! note

    Here, we use `skip_signature=True` because the data is public. We also need to set the cloud region for any data stored in AWS (this isn't required for all S3-compatible clouds).


=== "S3"

    ```python exec="on" session="usage" source="material-block"

    import xarray as xr
    from obstore.store import from_url

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    bucket = "s3://nex-gddp-cmip6"
    path = "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
    url = f"{bucket}/{path}"
    store = from_url(bucket, region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})

    ```

=== "GCS"

    ```python
    import xarray as xr
    from obstore.store import from_url

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    bucket = "gs://data-bucket"
    path = "file-path/data.nc"
    url = f"{bucket}/{path}"
    store = from_url(bucket)
    registry = ObjectStoreRegistry({bucket: store})

    ```

=== "Azure"

    ```python

    import xarray as xr
    from obstore.store import from_url

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    bucket = "abfs://data-container"
    path = "file-path/data.nc"
    url = f"{bucket}/{path}"
    store = from_url(bucket)
    registry = ObjectStoreRegistry({bucket: store})

    ```

=== "HTTP"

    ```python

    import xarray as xr
    from obstore.store import from_url

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    # This examples uses a NetCDF file of CMIP6 from ESGF.
    bucket  = 'https://esgf-data.ucar.edu'
    path = 'thredds/fileServer/esg_dataroot/CMIP6/CMIP/NCAR/CESM2/historical/r3i1p1f1/day/tas/gn/v20190308/tas_day_CESM2_historical_r3i1p1f1_gn_19200101-19291231.nc'
    store = from_url(bucket)
    registry = ObjectStoreRegistry({bucket: store})

    ```

=== "CEPH / OSN"

    ```python

    import xarray as xr
    from obstore.store import S3Store

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    endpoint = "https://nyu1.osn.mghpcc.org"
    access_key_id = "<access_key_id>"
    secret_access_key = "<secret_access_key>"
    path = "<path_to_files>"
    file = "filename.nc"
    scheme = "s3://"

    # create anon s3 store
    store = S3Store.from_url(f"{scheme}{path}", endpoint=endpoint, skip_signature=True)

    # create s3 store with aws-style credentials
    store = S3Store.from_url(f"{scheme}{path}", endpoint=endpoint, access_key_id = aws_access_key_id, secret_access_key=aws_secret_access_key)

    registry = ObjectStoreRegistry({f"{scheme}{path}": store})

    ```

=== "Local"

    ```python

    import xarray as xr
    from obstore.store import LocalStore

    from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
    from virtualizarr.parsers import HDFParser
    from virtualizarr.registry import ObjectStoreRegistry

    from pathlib import Path

    store_path = Path.cwd()
    file_path = str(store_path / "data.nc")
    file_url = f"file://{file_path}"

    store = LocalStore(prefix=store_path)
    registry = ObjectStoreRegistry({file_url: store})

    ```

Zarr can emit a lot of warnings about Numcodecs not being including in the Zarr version 3 specification yet -- let's suppress those.

```python exec="on" source="above" session="homepage"
import warnings
warnings.filterwarnings(
  "ignore",
  message="Numcodecs codecs are not in the Zarr version 3 specification*",
  category=UserWarning
)
```

We can open a virtual representation of this file using [virtualizarr.open_virtual_dataset][]. VirtualiZarr has various
"parsers" that understand different file formats. You must supply a parser, and as all netCDF4 files are HDF5 files,
here we used the [HDFParser][virtualizarr.parsers.HDFParser].

```python exec="on" session="usage" source="material-block"
parser = HDFParser()
vds = open_virtual_dataset(
  url=f"{bucket}/{path}",
  parser=parser,
  registry=registry,
)
```

!!! important

    It is good practice to use `open_virtual_dataset` as a context manager to automatically close file handles.
    For example the above code would become:

    ```python
    with vz.open_virtual_dataset('air.nc', registry=registry, parser=parser) as vds:
	    # do things with vds
        ...
    ```

    This is important to avoid accumulating open file handles and for avoiding leaks, so is recommended for production code.
    However we omit the context managers from the examples in the documentation for brevity.

Printing this "virtual dataset" shows that although it is an instance of [xarray.Dataset][], unlike a typical xarray dataset, it wraps [virtualizarr.manifests.ManifestArray][] objects in addition to a few in-memory NumPy arrays. You can learn more about the `ManifestArray` class in the [Data Structures documentation](data_structures.md).

```python exec="on" session="usage" source="material-block" result="code"
print(vds)
```

Generally a "virtual dataset" is any [xarray.Dataset][] which wraps one or more [virtualizarr.manifests.ManifestArray][] objects.

These particular [virtualizarr.manifests.ManifestArray][] objects are each a virtual reference to some data in the source NetCDF file, with the references stored in the form of [ChunkManifests][virtualizarr.manifests.ChunkManifest].

As the manifest contains only addresses at which to find large binary chunks, the virtual dataset takes up far less space in memory than the original dataset does:

```python exec="on" session="usage" source="material-block" result="code"
print(vds.nbytes)
```

```python exec="on" session="usage" source="material-block" result="code"
print(vds.vz.nbytes)
```

!!! important

    Virtual datasets are not normal xarray datasets!

    Although the top-level type is still `xarray.Dataset`, they are intended only as an abstract representation of a set of data files, not as something you can do analysis with.
    If you try to load, view, or plot any data you will get a `NotImplementedError`.
    Virtual datasets only support a very limited subset of normal xarray operations, particularly functions and methods for concatenating, merging and extracting variables, as well as operations for renaming dimensions and variables.

    _The only use case for a virtual dataset is [combining references](#combining-virtual-datasets) to files before [writing out those references to disk](#writing-virtual-stores-to-disk)._

## Loading variables

Once a virtual dataset is created, you won't be able to load the values of the virtual variables into memory.
Instead, you could load specific variables during virtual dataset creation using the `loadable_variables` parameter.
Loading the variables during virtual dataset creation has several benefits detailed in the [FAQ](faq.md#why-would-i-want-to-load-variables-using-loadable_variables).

```python exec="on" session="usage" source="material-block" result="code"
vds = open_virtual_dataset(
    url=url,
    registry=registry,
    parser=parser,
    loadable_variables=['time']
)
print(vds)
```

You can see that the dataset contains a mixture of virtual variables backed by `ManifestArray` objects (`tasmax`, `lat`, and `lon`), and loadable variables backed by (lazy) numpy arrays (`time`).

The default value of `loadable_variables` is `None`, which effectively specifies all the "dimension coordinates" in the file, i.e. all one-dimensional coordinate variables whose name is the same as the name of their dimensions. Xarray indexes will also be automatically created for these variables. Together these defaults mean that your virtual dataset will be opened with the same indexes as it would have been if it had been opened with just [xarray.open_dataset][].

!!! note
    In general, it is recommended to load all of your low-dimensional (e.g scalar and 1D) variables.

    Whilst this does mean the original data will be duplicated in your new virtual zarr store, by loading your coordinates into memory they can be inlined in the reference file or be stored as single chunks rather than large numbers of extremely tiny chunks, which speeds up loading that data on subsequent usages of the virtual dataset.

    However, you should not do this for much higher-dimensional variables, as then you might use a lot of storage duplicating them, defeating the point of the virtual zarr approach.

    Also, anything duplicated could become out of sync with the referenced original files, especially if not using a transactional storage engine such as `Icechunk`.

### Loading CF-encoded time variables

To decode time variables according to the CF conventions upon loading, you must ensure that variable is one of the `loadable_variables` and the `decode_times` argument of `open_virtual_dataset` is set to `True` (`decode_times` defaults to `None`).

```python exec="on" session="usage" source="material-block" result="code"
vds = open_virtual_dataset(
    url=url,
    registry=registry,
    parser=parser,
    loadable_variables=['time'],
    decode_times=True,
)
print(vds)
```

## Combining virtual datasets

In general we should be able to combine all the datasets from our archival files into one using some combination of calls to [xarray.concat][] and [xarray.merge][].
For combining along multiple dimensions in one call we also have [xarray.combine_nested][] and [xarray.combine_by_coords][].
If you're not familiar with any of these functions we recommend you skim through [xarray's docs on combining](https://docs.xarray.dev/en/stable/user-guide/combining.html).

!!! important
    Currently the virtual approach requires the same chunking and encoding across datasets. See the [FAQ](faq.md#can-my-specific-data-be-virtualized) for more details.

### Manual concatenation ordering

The simplest case of concatenation is when you have a set of files and you know the order in which they should be concatenated, _without looking inside the files_.
In this case it is sufficient to open the files one-by-one, then pass the virtual datasets as a list to the concatenation function.

```python exec="on" session="usage" source="material-block"
url_1 = "s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
url_2 = "s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2016_v2.0.nc"

vds1 = open_virtual_dataset(url=url_1, registry=registry, parser=parser)
vds2 = open_virtual_dataset(url=url_2, registry=registry, parser=parser)
```

As we know the correct order a priori, we can just combine along one dimension using [xarray.concat][].

```python exec="on" session="usage" source="material-block" result="code"
combined_vds = xr.concat([vds1, vds2], dim='time')
print(combined_vds)
```

!!! note
    If you have any virtual coordinate variables, you will likely need to specify the keyword arguments `coords='minimal'` and `compat='override'` to `xarray.concat()`, because the default behaviour of xarray will attempt to load coordinates in order to check their compatibility with one another.  Similarly, if there are data variables that do not include the concatenation dimension, you will likely need to specify `data_vars='minimal'`.

    In future this [default will be changed](https://github.com/pydata/xarray/issues/8778), such that passing these two arguments explicitly will become unnecessary.

The general multi-dimensional version of this concatenation-by-order-supplied can be achieved using `xarray.combine_nested()`.

```python exec="on" session="usage" source="material-block"
combined_vds = xr.combine_nested([vds1, vds2], concat_dim=['time'])
```

In N-dimensions the datasets would need to be passed as an N-deep nested list-of-lists, see the [xarray docs](https://docs.xarray.dev/en/stable/user-guide/combining.html#combining-along-multiple-dimensions).

!!! note
    For manual concatenation we can actually avoid creating any xarray indexes, as we won't need them.
    Without indexes we can avoid loading any data whatsoever from the files.
    However, you should first be confident that the archival files actually do have compatible data, as the coordinate values then cannot be efficiently compared for consistency (i.e. aligned).

You can achieve both the opening and combining steps for multiple files in one go by using [open_virtual_mfdataset][virtualizarr.open_virtual_mfdataset].

```python exec="on" session="usage" source="material-block"
combined_vds = open_virtual_mfdataset(
    [url_1, url_2],
    registry=registry,
    parser=parser,
    combine="nested",
    concat_dim="time"
)
```

We passed `combine='nested'` to specify that we want the datasets to be combined in the order they appear, using `xr.combine_nested` under the hood.

### Ordering by coordinate values

If you're happy to load 1D dimension coordinates into memory, you can use their values to do the ordering for you!

```python exec="1" session="usage" source="material-block"
vds1 = open_virtual_dataset(url=url_1, registry=registry, parser=parser, loadable_variables=['time','lat', 'lon'], decode_times=True)
vds2 = open_virtual_dataset(url=url_2, registry=registry, parser=parser, loadable_variables=['time','lat', 'lon'], decode_times=True)

combined_vds = xr.combine_by_coords([vds2, vds1], combine_attrs="drop_conflicts")
```

Notice we don't have to specify the concatenation dimension explicitly - xarray works out the correct ordering for us.
Even though we actually passed in the virtual datasets in the wrong order just now, they have been combined in the correct order such that the 1-dimensional `time` coordinate has ascending values.
As a result our virtual dataset still has the data in the correct order.

```python exec="on" session="usage" source="material-block" result="code"
print(combined_vds)
```

Again, we can achieve both the opening and combining steps for multiple files in one go by using [open_virtual_mfdataset][virtualizarr.open_virtual_mfdataset], but this passing `combine='by_coords'`.

```python exec="on" session="usage" source="material-block"
combined_vds = open_virtual_mfdataset(
    [url_1, url_2],
    registry=registry,
    parser=parser,
    combine="by_coords",
    combine_attrs="drop_conflicts",
)
```

In the future, we aim to provide globbing utilities to simplify finding datasets to include.

### Ordering using metadata

You can create a new index from the url by passing a function to the `preprocess` parameter of [open_virtual_mfdataset][virtualizarr.open_virtual_mfdataset]. An example will be added in the future.

### Combining many virtual datasets at once

Combining a large number (e.g., 1000s) of virtual datasets at once should be very quick (a few seconds), as we are manipulating only a few KBs of metadata in memory.

However creating 1000s of virtual datasets at once can take a very long time.
(If it were quick to do so, there would be little need for this library!)
See the page on [Scaling](scaling.md) for tips on how to create large numbers of virtual datasets at once.

## Changing the prefix of urls in the virtual dataset

You can update the urls stored in a manifest or virtual dataset without changing the byte range information using the [virtualizarr.VirtualiZarrDatasetAccessor.rename_paths][] accessor method.

For example, you may want to rename urls according to a function to reflect having moved the location of the referenced files from local storage to an S3 bucket.

```python
def local_to_s3_url(old_local_path: str) -> str:
    from pathlib import Path

    new_s3_bucket_url = "http://s3.amazonaws.com/my_bucket/"

    filename = Path(old_local_path).name
    return str(new_s3_bucket_url / filename)

renamed_vds = vds.vz.rename_paths(local_to_s3_url)
```

## Writing virtual stores to disk

Once we've combined references to all the chunks of all our archival files into one virtual xarray dataset, we still need to store those references so that they can be read by our analysis code later.


### Writing to an Icechunk Store

We can store these references using [Icechunk](https://icechunk.io/).
`Icechunk` is an open-source, cloud-native transactional tensor storage engine that is fully compatible with Zarr-Python version 3, as it conforms to the Zarr V3 specification.
To export our virtual dataset to an `Icechunk` Store, we use the [virtualizarr.VirtualiZarrDatasetAccessor.to_icechunk][] accessor method.

Here we use a memory store but in real use-cases you'll probably want to use [icechunk.local_filesystem_storage][], [icechunk.s3_storage][], [icechunk.azure_storage][], [icechunk.gcs_storage][], or a similar storage class.

```python exec="on" session="usage" source="material-block" result="code"
import icechunk

# you need to explicitly grant permissions to icechunk to read from the locations of your archival files
# we use `anonymous=True` because this is a public bucket, otherwise you need to set credentials explicitly
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix="s3://nex-gddp-cmip6/",
        store=icechunk.s3_store(region="us-west-2", anonymous=True),
    ),
)

# create an in-memory icechunk repository that includes the virtual chunk containers
storage = icechunk.in_memory_storage()
repo = icechunk.Repository.create(storage, config)

# open a writable icechunk session to be able to add new contents to the store
session = repo.writable_session("main")

# write the virtual dataset to the session's IcechunkStore instance, using VirtualiZarr's `.vz` accessor
vds1.vz.to_icechunk(session.store)

# commit your changes so that they are permanently available as a new snapshot
snapshot_id = session.commit("Wrote first dataset")
print(snapshot_id)

# optionally persist the new permissions to be permanent, which you probably want
# otherwise every user who wants to read the referenced virtual data back later will have to repeat the `config.set_virtual_chunk_container` step at read time.
repo.save_config()
```

#### Append to an existing Icechunk Store

You can append a virtual dataset to an existing Icechunk store using the `append_dim` argument.
This option is designed to behave similarly to the `append_dim` option to xarray's [xarray.Dataset.to_zarr][] method, and is especially useful for datasets that grow over time.

!!! important
    Note again that the virtual Zarr approach requires the same chunking and encoding across datasets. This including when appending to an existing Icechunk-backed Zarr store. See the [FAQ](faq.md#can-my-specific-data-be-virtualized) for more details.

```python exec="on" session="usage" source="material-block" result="code"
# write the virtual dataset to the session with the IcechunkStore
session = repo.writable_session("main")
vds2.vz.to_icechunk(session.store, append_dim="time")
snapshot_id = session.commit("Appended second dataset")
print(snapshot_id)
```

See the [Icechunk documentation](https://icechunk.io/en/latest/virtual/) for more details.

### Writing to Kerchunk's format and reading data via fsspec

The [kerchunk library](https://github.com/fsspec/kerchunk) has its own [specification](https://fsspec.github.io/kerchunk/spec.html) for serializing virtual datasets as a JSON file or Parquet directory.

To write out all the references in the virtual dataset as a single kerchunk-compliant JSON or parquet file, you can use the [virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk][] accessor method.

```python
combined_vds.vz.to_kerchunk('output/combined.json', format='json')
```

These zarr-like references can now be interpreted by [fsspec](https://github.com/fsspec/filesystem_spec), using kerchunk's built-in xarray backend (kerchunk must be installed to use `engine='kerchunk'`).

```python
combined_ds = xr.open_dataset('output/combined.json', engine="kerchunk")
print(combined_ds)
```

In-memory ("loadable") variables backed by numpy arrays can also be written out to kerchunk reference files, with the values serialized as bytes.
This is equivalent to kerchunk's concept of "inlining", but done on a per-array basis using the `loadable_variables` kwarg rather than a per-chunk basis using kerchunk's `inline_threshold` kwarg.

!!! note
    Currently you can only serialize in-memory variables to kerchunk references if they do not have any encoding.

When you have many chunks, the reference file can get large enough to be unwieldy as JSON.
In that case the references can be instead stored as parquet, which again this uses kerchunk internally.

```python
combined_vds.vz.to_kerchunk('output/combined.parquet', format='parquet')
```

And again we can read these references using the "kerchunk" backend

```python
combined_ds = xr.open_dataset('output/combined.parquet', engine="kerchunk")
print(combined_ds)
```

By default references are placed in separate parquet files when the total number of references exceeds `record_size`.
If there are fewer than `categorical_threshold` unique urls referenced by a particular variable, url will be stored as a categorical variable.

## Opening Kerchunk references as virtual datasets

You can open existing Kerchunk `json` or `parquet` references as VirtualiZarr virtual datasets.
This may be useful for manipulating them or converting existing kerchunk-formatted references to other reference storage formats such as [Icechunk](https://icechunk.io/).

```python
from pathlib import Path
from virtualizarr.parsers import KerchunkJSONParser, KerchunkParquetParser

url_cwd = f"file://{str(Path.cwd())}"
store = from_url(url_cwd)
registry.register(url_cwd, store)
vds = open_virtual_dataset('output/combined.json', registry=registry, parser=KerchunkJSONParser())
# or
vds = open_virtual_dataset('output/combined.parquet', registry=registry, parser=KerchunkParquetParser())
```

One difference between the kerchunk references format and virtualizarr's internal manifest representation (as well as Icechunk's format) is that paths in kerchunk references can be relative paths.

Opening kerchunk references that contain relative local filepaths therefore requires supplying another piece of information: the directory of the `fsspec` filesystem which the filepath was defined relative to.

You can dis-ambuiguate kerchunk references containing relative paths by passing the `fs_root` kwarg to `virtual_backend_kwargs`.

```python
# file `relative_refs.json` contains a path like './file.nc'

vds = open_virtual_dataset(
    'relative_refs.json',
    registry=registry,
    parser=KerchunkJSONParser(
        fs_root='file:///data_directory/',
    )
)

# the path in the virtual dataset will now be 'file:///data_directory/file.nc'
```

Note that as the virtualizarr [virtualizarr.VirtualiZarrDatasetAccessor.to_kerchunk][] method only writes absolute paths, the only scenario in which you might come across references containing relative paths is if you are opening references that were previously created using the `kerchunk` library alone.
