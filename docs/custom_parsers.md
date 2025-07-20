# Custom parsers

This page explains how to write a custom parser for VirtualiZarr, to extract chunk references from an archival data format not already supported by the main package.
This is advanced material intended for 3rd-party developers, and assumes you have read the page on [Data Structures](data_structures.md).

!!! note
    "Parsers" were previously known variously as "readers" or "backends" in older versions of VirtualiZarr.
    We renamed them to avoid confusion with obstore readers and xarray backends.

## What is a VirtualiZarr parser?

All VirtualiZarr parsers are simply callables that accept the URL pointing to a data source and a [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] that may contain instantiated [ObjectStores][obstore.store.ObjectStore] that can read from that URL, and return an instance of the [`virtualizarr.manifests.ManifestStore`][] class containing information about the contents of the data source.

```python
from virtualizarr.manifests import ManifestStore
from virtualizarr.registry import ObjectStoreRegistry


def custom_parser(url: str, registry: ObjectStoreRegistry) -> ManifestStore:
    # access the file's contents, e.g. using the ObjectStore instance in the registry
    store, path_in_store = registry.resolve(url)
    readable_file = obstore.open_reader(store, path_in_store)

    # parse the file contents to extract its metadata
    # this is generally where the format-specific logic lives
    manifestgroup: ManifestGroup = extract_metadata(readable_file)

    # construct the Manifeststore from the parsed metadata and the object store registry
    return ManifestStore(group=manifestgroup, registry=registry)


vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=custom_parser,
)
```

All parsers _must_ follow this exact call signature, enforced at runtime by checking against the [`virtualizarr.parsers.typing.Parser`][] typing protocol.

!!! note
    The object store registry can technically be empty, but to be able to read actual chunks of data back from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] later, the registry needs to contain at least one [`ObjectStore`][obstore.store.ObjectStore] matched to the URL prefix of the data sources.

    The only time you might want to use an empty object store registry is if you are attempting to parse a custom metadata-only references format without touching the original files they refer to -- i.e., a format like Kerchunk or DMR++, that doesn't contain actual binary data values.

## What is the responsibility of a parser?

The VirtualiZarr package really does four separate things, in order:

1. Maps the contents of common archival file formats to the Zarr data model, including references to the locations of the chunks.
2. Allows reading chosen variables into memory (e.g. via the `loadable_variables` kwarg, or reading from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] using zarr-python directly).
3. Provides a way to combine arrays of chunk references using a convenient API (the Xarray API).
4. Allows persisting these references to storage for later use, in either the Kerchunk or Icechunk format.

**VirtualiZarr parsers are responsible for the entirety of step (1).**
In other words, all of the assumptions required to map the data model of an archival file format to the Zarr data model, and the logic for doing so for a specific file, together constitute a parser.

**The ObjectStore instances are responsible for fetching the bytes in step (2).**

This design provides a neat separation of concerns, which is helpful in two ways:

1. The Xarray data model is subtly different from the Zarr data model (see below), so as the final objective is to create a virtual store which programmatically maps Zarr API calls to the archival file format at read-time, it is useful to separate that logic up front, before we convert to use the xarray virtual dataset representation and potentially subtly confuse matters.
2. It also allows us to support reading data from the file via the [`ManifestStore`][virtualizarr.manifests.ManifestStore] interface, using zarr-python and obstore, but without using Xarray.

## Reading data from the `ManifestStore`

As well as being a well-defined representation of the archival data in the Zarr model, you can also read chunk data directly from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] object.

This works because the [`ManifestStore`][virtualizarr.manifests.ManifestStore] class is an implementation of the Zarr-Python `zarr.abc.Store` interface, and uses the [obstore](https://developmentseed.org/obstore/latest/) package internally to actually fetch chunk data when requested.

Reading data from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] can therefore be done using the zarr-python API directly:

```python
manifest_store = parser(url, registry)

zarr_group = zarr.open_group(manifest_store)
zarr_group.tree()
```
or using xarray:
```python
manifest_store = parser(url, registry)

ds = xr.open_zarr(manifest_store, zarr_format=3, consolidated=False)
```

Note using xarray like this would produce an entirely non-virtual dataset, so is equivalent to passing

```python
ds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
    loadable_variables=<all_the_variable_names>,
)
```

## How is the parser called internally?

The parser is passed to [`open_virtual_dataset`][virtualizarr.open_virtual_dataset], and immediately called on the url to produce a [`ManifestStore`][virtualizarr.manifests.ManifestStore] instance.

The [`ManifestStore`][virtualizarr.manifests.ManifestStore] is then converted to the xarray data model using [`ManifestStore.to_virtual_dataset`][virtualizarr.manifests.ManifestStore.to_virtual_dataset], which loads `loadable_variables` by reading from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] using [`xarray.open_zarr`][].

This virtual dataset object is then returned to the user, so [`open_virtual_dataset`][virtualizarr.open_virtual_dataset] is really a very thin wrapper around the parser.

## Parser-specific keyword arguments

The [`Parser`][virtualizarr.parsers.typing.Parser] `__call__` method does not accept arbitrary optional keyword arguments.

However, extra information is often needed to fully map the archival format to the Zarr data model, for example if the format does not include array names or dimension names.

Instead, to pass arbitrary extra information to your parser callable, it is recommended that you bind that information to class attributes (or use [`functools.partial`][]). For example:

```python
class CustomParser:
    def __init__(self, option_1: bool = False,  option_2: int | None = None) -> None:
        self.option_1 = option_1
        self.option_2 = option_2

    def __call__(self, url: str, registry: ObjectStoreRegistry) -> ManifestStore:
        # access the file's contents, e.g. using the ObjectStore instance in the registry
        store, path_in_store = registry.resolve(url)
        readable_file = obstore.open_reader(store, path_in_store)

        # parse the file contents to extract its metadata
        # this is generally where the format-specific logic lives
        manifestgroup: ManifestGroup = extract_metadata(readable_file, option_1=self.option_1, option_2=self.option_2)

        # construct the Manifeststore from the parsed metadata and the object store registry
        return ManifestStore(group=manifestgroup, registry=registry)

parser = CustomParser(option_1=True, option_2=6)
vds = vz.open_virtual_dataset(
    url,
    registry=registry,
    parser=parser,
)
```

This helps to keep format-specific parser configuration separate from kwargs to [`open_virtual_dataset`][virtualizarr.open_virtual_dataset].

## How to write your own custom parser

As long as your custom parser callable follows the interface above, you can implement it in any way you like.
However there are few common approaches.

### Typical VirtualiZarr parsers

The recommended way to implement a custom parser by parsing a file of a given format, and construct the [`ManifestStore`][virtualizarr.manifests.ManifestStore] object explicitly, component by component, extracting the metadata that you need.

Generally you want to follow steps like this:

1.  Extract file header or magic bytes to confirm the file passed is the format your parser expects.
2.  Read metadata to determine how many arrays there are in the file, their shapes, chunk shapes, dimensions, codecs, and other metadata.
3.  For each array in the file:
    1.  Create a `zarr.core.metadata.ArrayV3Metadata` object to hold that metadata, including dimension names. At this point you may have to define new Zarr codecs to support deserializing your data (though hopefully the standard Zarr codecs are sufficient).
    2.  Extract the byte ranges of each chunk and store them alongside the fully-qualified filepath in a [`ChunkManifest`][virtualizarr.manifests.ChunkManifest] object.
    3. Create one [`ManifestArray`][virtualizarr.manifests.ManifestArray] object, using the corresponding `zarr.core.metadata.ArrayV3Metadata` and [`ChunkManifest`][virtualizarr.manifests.ChunkManifest] objects.
4.  Group [`ManifestArray`][virtualizarr.manifests.ManifestArray]s into one or more [`ManifestGroup`][virtualizarr.manifests.ManifestGroup] objects. Ideally you would only have one group, but your format's data model may preclude that. If there is group-level metadata attach this to the [`ManifestGroup`][virtualizarr.manifests.ManifestGroup] object as a `zarr.metadata.GroupMetadata` object. Remember that [`ManifestGroup`][virtualizarr.manifests.ManifestGroup]s can contain other groups as well as arrays.
5.  Instantiate the final [`ManifestStore`][virtualizarr.manifests.ManifestStore] using the top-most [`ManifestGroup`][virtualizarr.manifests.ManifestGroup] and return it.

!!! note

    The [regular chunk grid](https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/chunk-grids/regular-grid/index.rst) for Zarr V3 data expects that chunks at the border of an array always have the full chunk size, even when the array only covers parts of it.

    For example, having an array with ``"shape": [30, 30]`` and ``"chunk_shape": [16, 16]``, the chunk ``0,1`` would also contain unused values for the indices ``0-16, 30-31``. If the file format that you are virtualizing does not fill in partial chunks, it is recommended that you raise a `ValueError` until Zarr supports [variable chunk sizes](https://github.com/orgs/zarr-developers/discussions/52).

### Parsing existing in-memory references

A `Parser` can be used to parse existing references that are in-memory, as well as those on-disk. This can be done by using the [`obstore.store.MemoryStore`][], and highlights the power of delegating all IO to a general store-like interface.

```python
# some example kerchunk-formatted JSON references, as an in-memory python dict
refs = {
    'version': 1,
    'refs': {
        '.zgroup': '{"zarr_format":2}',
        'a/.zarray': '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
        'a/.zattrs': '{"_ARRAY_DIMENSIONS":["x","y"],"value": "1"}',
        'a/0.0': ['/test1.nc', 6144, 48],
    }
}

memory_store = obstore.store.MemoryStore()
memory_store.put("refs.json", ujson.dumps(refs).encode())

registry = ObjectStoreRegistry({"memory://": memory_store})
parser = KerchunkJSONParser()
manifeststore = parser("file:///Users/user/my-project/refs.json", registry)
```

Note that the [`MemoryStore`][obstore.store.MemoryStore] is needed for reading metadata(/inlined chunk data) from the in-memory `dict`, but if you wanted to be able to load data actually referred to by the kerchunk references you would need more stores in the registry (for example to read from the local file `/test1.nc` the registry would also need to contain a [`LocalStore`][obstore.store.LocalStore]).

### Parsing a sidecar file

A custom parser can parse multiple files, perhaps by passing a glob string and looking for expected file naming conventions, or by passing additional parser-specific keyword arguments.
This can be useful for reading file formats which include some kind of additional "index" sidecar file, but don't have all the information necessary to construct the entire [`ManifestStore`][virtualizarr.manifests.ManifestStore] object from the sidecar file alone.

!!! note
    If you do have some type of custom sidecar metadata file which contains all the information necessary to create the [`ManifestStore`][virtualizarr.manifests.ManifestStore], then you should just create a custom parser for that metadata file format instead!
    Examples of this approach which come packaged with VirtualiZarr are the [`DMRPPparser`][virtualizarr.parsers.DMRPPParser] and the [`KerchunkJSONparser`][virtualizarr.parsers.KerchunkJSONParser]

### Kerchunk-based parsers

The Kerchunk package includes code for parsing various array file formats, returning the result as an in-memory nested dictionary, following the [Kerchunk references specification](https://fsspec.github.io/kerchunk/spec).
These references can be directly read and converted into a [`ManifestStore`][virtualizarr.manifests.ManifestStore] by VirtualiZarr's [`KerchunkJSONParser`][virtualizarr.parsers.KerchunkJSONParser] and [`KerchunkParquetParser`][virtualizarr.parsers.KerchunkParquetParser].

You can therefore use a function which returns in-memory kerchunk JSON references inside your parser, then simply call [`KerchunkJSONParser`][virtualizarr.parsers.KerchunkJSONParser] and return the result.

!!! note
    Whilst this might be the quickest way to get a custom parser working, we do not really recommend this approach, as:

    1. The Kerchunk in-memory nested dictionary format is very memory-inefficient compared to the numpy array representation used internally by VirtualiZarr's [`ChunkManifest`][virtualizarr.manifests.ChunkManifest] class,
    2. The Kerchunk package in general has a number of known bugs, often stemming from a lack of clear internal abstractions and specification,
    3. This lack of data model enforcement means that the dictionaries returned by different Kerchunk parsers sometimes follow inconsistent schemas ([for example](https://github.com/fsspec/kerchunk/issues/561)).

    Nevertheless this approach is used by VirtualiZarr internally, at least for the FITS, netCDF3, and the (since-deprecated-and-removed original implementation of the) HDF5 file format parsers.

## Data model differences between Zarr and Xarray

Whilst the [`ManifestStore`][virtualizarr.manifests.ManifestStore] class enforces nothing other than the minimum required to conform to the Zarr model, if you want to convert your [`ManifestStore`][virtualizarr.manifests.ManifestStore] to a virtual xarray dataset using [`ManifestStore.to_virtual_dataset`][virtualizarr.manifests.ManifestStore.to_virtual_dataset], there are a couple of additional requirements, set by Xarray's data model.

1. All arrays must have dimension names, specified in the `zarr.core.metadata.ArrayV3Metadata` objects.
2. All arrays in the same group with a common dimension name must have the same length along that common dimension.

You also may want to set the `coordinates` field of the group metadata to tell xarray to set those variables as coordinates upon conversion.

## Testing your new parser

The fact we can read data from the [`ManifestStore`][virtualizarr.manifests.ManifestStore] is useful for testing that our parser implementation behaves as expected.

If we already have some other way to read data directly into memory from that archival file format -- for example, a conventional xarray IO backend -- we can compare the results of opening and loading data via the two approaches.

For example we could test the ability of VirtualiZarr's in-built [`HDFParser`][virtualizarr.parsers.HDFParser] to read netCDF files by comparing the output to xarray's `h5netcdf` backend.

```python
import xarray.testing as xrt

from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from obstore.store import LocalStore

project_directory = "/Users/user/my-project"
project_url = f"file://{project_directory}"
registry = ObjectStoreRegistry({project_url: LocalStore(prefix=project_directory)})
parser = HDFParser()
manifest_store = parser(url=f"{project_url}/netcdf-file.nc", registry=registry)

with (
    xr.open_dataset(manifest_store, engine="zarr", zarr_format=3, consolidated=False) as actual,
    xr.open_dataset(manifest_store, backend="h5netcdf") as expected,
):
    xrt.assert_identical(actual, expected)
```

These two approaches do not share any IO code, other than potentially the CF-metadata decoding that [`xarray.open_dataset`][] optionally applies when opening any file.
Therefore if the results are the same, we know our custom parser implementation behaves as expected, and that reading the netCDF data back via Icechunk/Kerchunk should give the same result as reading it directly.
