# Custom Readers

This page explains how to write a custom reader for VirtualiZarr, to extract chunk references from an archival data format not already supported by the main package.
This is advanced material intended for 3rd-party developers, and assumes you have read the page on [Data Structures](data_structures.md).

## What is a VirtualiZarr reader?

All VirtualiZarr readers are simply functions that accept a path to a file of a specific format, and return an instance of the `ManifestStore` class containing information about the contents of that file.

```python
from virtualizarr.manifests import ManifestStore

def a_custom_reader(path: str, **kwargs) -> ManifestStore:
    # parse the file contents here

    # construct the Manifeststore
    manifest_store = ManifestStore(...)

    return manifest_store
```

## What is the responsibility of a reader?

The VirtualiZarr package really does four separate things.
In order, it:

1. Maps the contents of common archival file formats to the Zarr data model, including references to the locations of the chunks.
2. Loads chosen variables into memory (the `loadable_variables`).
3. Provides a way to combine arrays of chunk references using a convenient API (the Xarray API).
3. Allows persisting these references to storage for later use, in either the Kerchunk or Icechunk format.

**VirtualiZarr readers are responsible for the entirety of step (1).**
In other words, all of the assumptions required to map the data model of an archival file format to the Zarr data model, and the logic for doing so for a specific file, together constitute a reader.

This design provides a neat separation of concerns, which is helpful in two ways:
1. The Xarray data model is subtly different from the Zarr data model (see below), so as the final objective is to create a virtual store which programmatically maps Zarr API calls to the archival file format at read-time, it is useful to separate that logic up front, before we convert to use the xarray virtual dataset representation and potentially subtly confuse matters.
2. It also allows us to support reading data from the file via the `ManifestStore` interface, without using Xarray directly.

## Reading data from the `ManifestStore`

As well as being a well-defined representation of the archival data in the Zarr model, you can also read chunk data directly from the `ManifestStore` object.

This works because the `ManifestStore` class is an implementation of the Zarr-Python `zarr.abc.Store` interface, and uses the `obstore` package internally to actually fetch chunk data when requested.

Reading data from the `ManifestStore` can therefore be done using the zarr-python API directly, or using xarray:

```python
ds = xr.open_zarr(manifest_store)
# or
ds = xr.open_dataset(manifest_store, engine='zarr')
```

This would be produce an entirely non-virtual dataset, so is equivalent to passing

```python
ds = vz.open_virtual_dataset(manifest_store, loadable_variables=<all_the_variable_names>)
```

## How is the reader function called internally?

The reader function is passed to `open_virtual_dataset`, and immediately called on the filepath to produce a `ManifestStore` instance.

The `ManifestStore` is then converted to the xarray data model using `Manifeststore.to_virtual_dataset()`, which loads `loadable_variables` by reading from the `ManifestStore` using `xr.open_zarr`.

This virtual dataset object is then returned to the user, so `open_virtual_dataset` is really a very thin wrapper around the reader function you pass.

## Reader-specific keyword arguments

Reader functions also accept arbitrary optional keyword arguments.
These are useful particularly to pass any extra information needed to fully map the archival format to the Zarr data model, for example if the format does not include array names or dimension names.

## How to write your own custom reader

As long as your custom reader function follows the interface above, you can implement it in any way you like.
However there are few common approaches.

### Typical VirtualiZarr readers

The recommended way to implement a custom reader is simply to parse the given file yourself, and construct the `ManifestStore` object explicitly component by component.

Generally you want to follow steps like this:
1. Extract file header or magic bytes to confirm the file passed is the format your reader expects.
2. Read metadata to determine how many arrays there are in the file, their shapes, chunk shapes, dimensions, codecs, and other metadata.
3. For each array in the file:
  4. Create a `zarr.core.metadata.ArrayV3Metadata` object to hold that metadata, including dimension names. At this point you may have to define new Zarr codecs to support deserializing your data (though hopefully the standard Zarr codecs are sufficient).
  5. Extract the byte ranges of each chunk and store them alongside the fully-qualified filepath in a `ChunkManifest` object.
  6. Create one `ManifestArray` object, using the corresponding `ArrayV3Metadata` and `ChunkManifest` objects.
7. Group `ManifestArrays` up into one or more `ManifestGroup` objects. Ideally you would only have one group, but your format's data model may preclude that. If there is group-level metadata attach this to the `ManifestGroup` object as a `zarr.metadata.GroupMetadata` object. Remember that `ManifestGroups` can contain other groups as well as arrays.
8. Instantiate the final `ManifestStore` using the top-most `ManifestGroup` and return it.

!!! note
  The [regular chunk grid](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/chunk-grids/regular-grid/index.rst) for Zarr V3 data expects that chunks at the border of an array always have the full chunk size, even when the array only covers parts of it. For example, having an array with ``"shape": [30, 30]`` and ``"chunk_shape": [16, 16]``, the chunk ``0,1`` would also contain unused values for the indices ``0-16, 30-31``. If the file format that you are virtualizing does not fill in partial chunks, it is recommended that you raise a `ValueError` until Zarr supports [variable chunk sizes](https://github.com/orgs/zarr-developers/discussions/52).

### Parsing a pre-existing index file

A custom reader can parse multiple files, perhaps by passing a glob string and looking for expected file naming conventions, or by passing additional reader-specific keyword arguments.
This can be useful for reading file formats which include some kind of additional "index" sidecar file, but don't have all the information necessary to construct the entire `ManifestStore` object from the sidecar file alone.

!!! note
  If you do have some type of custom sidecar metadata file which contains all the information necessary to create the `ManifestStore`, then you should just create a custom reader for that metadata file format instead!
  Examples of this approach which come packaged with VirtualiZarr are the `DMRPPReader` and the `KerchunkReader`

### Kerchunk-based readers

The Kerchunk package includes code for parsing various array file formats, returning the result as an in-memory nested dictionary objects, following the [Kerchunk references specification](https://fsspec.github.io/kerchunk/spec).
These references can be directly read and converted into a `ManifestStore` by VirtualiZarr's `KerchunkReader`.

!!! note
  Whilst this might be the quickest way to get a custom reader working, we do not really recommend this approach, as:
  1. The Kerchunk in-memory nested dictionary format is very memory-inefficient compared to the numpy array representation used internally by VirtualiZarr's `ChunkManifest` class,
  2. The Kerchunk package in general has a number of known bugs, often stemming from a lack of clear internal abstractions and specification,
  3. This lack of data model enforcement means that the dictionaries returned by different Kerchunk readers sometimes follow inconsistent schemas ([an example]()).

  Nevertheless this approach is currently used by VirtualiZarr internally, at least for the FITS, netCDF3, and (now-deprecated original implementation of the) HDF5 file format readers.

## Data model differences between Zarr and Xarray

Whilst the `ManifestStore` class enforces nothing other than the minimum required to conform to the Zarr model, if you want to convert your `ManifestStore` to a virtual xarray dataset using `ManifestStore.to_virtual_dataset()`, there are a couple of additional requirements, set by Xarray's data model.

1. All arrays must have dimension names, specified in the `ArrayV3Metadata` objects.
2. All arrays in the same group with a common dimension name must have the same length along that common dimension.

You also may want to set the `coordinates` field of the group metadata to tell xarray to set those variables as coordinates upon conversion.

## Testing your new reader

The fact we can read data from the `ManifestStore` is useful for testing that our reader implementation behaves as expected.

If we already have some other way to read data directly into memory from that archival file format - for example a conventional xarray IO backend - we can compare the results of opening and loading data via the two approaches.

For example we could test the ability of VirtualiZarr's in-built `HDFBackend` to read netCDF files by comparing the output to xarray's `h5netcdf` backend.

```python
import xarray.testing as xrt

from virtualizarr.readers import HDFBackend

manifest_store = HDFBackend("file.nc")
actual = xr.open_dataset(manifest_store, engine="zarr")

expected = xr.open_dataset(manifest_store, backend="h5netcdf")
xrt.assert_identical(actual, expected)
```

These two approaches do not share any IO code, other than potentially the CF-metadata decoding that `xarray.open_dataset` optionally applies when opening any file.
Therefore if the results are the same, we know our custom reader implementation behaves as expected, and that reading the netCDF data back via Icechunk/Kerchunk should give the same result as reading it directly.
