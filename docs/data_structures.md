# Data structures

This page explains how VirtualiZarr works, by introducing the core data structures one-by-one.

## Chunk Manifests

In the Zarr model N-dimensional arrays are stored as a series of compressed chunks, each labelled by a chunk key which indicates its position in the array.
Whilst conventionally each of these Zarr chunks are a separate compressed binary file stored within a Zarr Store, there is no reason why these chunks could not actually already exist as part of another file (e.g. a netCDF file), and be loaded by reading a specific byte range from this pre-existing file.

A "Chunk Manifest" is a list of chunk keys and their corresponding byte ranges in specific files, grouped together such that all the chunks form part of one Zarr-like array.
For example, a chunk manifest for a 3-dimensional array made up of 4 chunks from the same file might look like this:

```python
{
    "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
    "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
    "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
    "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
}
```

Notice that in this case the `"path"` attributes all point to a single a netCDF file `"foo.nc"` stored in a remote S3 bucket.
A single chunk manifest can store references to any number of chunks, spread across any number of files, in any number of locations.

Note there is no need for the files the chunk manifest refers to to be local, or even to be currently accessible to your code (but you will need to be able to access them when you intend to read the actual chunk data!).

The virtual dataset we created in the [usage guide](usage.md) above contains multiple chunk manifests stored in-memory, which we can see by pulling one out as a python dictionary.

```python
marr = vds['air'].data
manifest = marr.manifest
manifest.dict()
```

```python
{'0.0.0': {'path': 'file:///work/data/air.nc', 'offset': 15419, 'length': 7738000}}
```

In this case we can see that the `"air"` variable contains only one chunk, the bytes for which live in the `file:///work/data/air.nc` file, at the location given by the `'offset'` and `'length'` attributes.

The [virtualizarr.manifests.ChunkManifest][] class is virtualizarr's internal in-memory representation of this manifest.

## `ManifestArray` class

A Zarr array is defined not just by the location of its constituent chunk data, but by its array-level attributes such as `shape` and `dtype`.
The [virtualizarr.manifests.ManifestArray][] class stores both the array-level attributes and the corresponding chunk manifest.

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
marr.metadata
```

```
ArrayV3Metadata(shape=(2920, 25, 53),
                data_type=<DataType.float64: 'float64'>,
                chunk_grid=RegularChunkGrid(chunk_shape=(2920, 25, 53)),
                chunk_key_encoding=DefaultChunkKeyEncoding(name='default',
                                                           separator='/'),
                fill_value=np.float64(-327.67),
                codecs=(FixedScaleOffset(codec_name='numcodecs.fixedscaleoffset', codec_config={'scale': 100.0, 'offset': 0, 'dtype': '<f8', 'astype': '<i2'}),
                        BytesCodec(endian=<Endian.little: 'little'>)),
                attributes={'GRIB_id': 11,
                            'GRIB_name': 'TMP',
                            'actual_range': [185.16000366210938,
                                             322.1000061035156],
                            'dataset': 'NMC Reanalysis',
                            'level_desc': 'Surface',
                            'long_name': '4xDaily Air temperature at sigma '
                                         'level 995',
                            'parent_stat': 'Other',
                            'precision': 2,
                            'statistic': 'Individual Obs',
                            'units': 'degK',
                            'var_desc': 'Air temperature'},
                dimension_names=None,
                zarr_format=3,
                node_type='array',
                storage_transformers=())
```

A `ManifestArray` can therefore be thought of as a virtualized representation of a single Zarr array.

As it defines various array-like methods, a `ManifestArray` can often be treated like a ["duck array"](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html) - i.e. other libraries can treat it as an multidimensional array without special casing for its type or content.
In particular, concatenation of multiple `ManifestArray` objects can be done via merging their chunk manifests into one (including automatic re-labelling of their chunk keys).

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

This concatenation property is what allows us to combine the data from multiple files on disk into a single Zarr store containing arrays of many chunks.

!!! note
    As a single Zarr array has only one array-level set of compression codecs by definition, concatenation of arrays from files saved to disk with differing codecs cannot be achieved through concatenation of `ManifestArray` objects.

    Implementing this feature will require a more abstract and general notion of concatenation, see [GH issue #5](https://github.com/zarr-developers/VirtualiZarr/issues/5). See the [FAQ](faq.md#can-my-specific-data-be-virtualized) for other restrictions on what data can be virtualized.

Remember that you cannot load values from a `ManifestArray` directly.

```python
vds['air'].values
```

```python
NotImplementedError: ManifestArrays can't be converted into numpy arrays or pandas Index objects
```

The whole point is to manipulate references to the data without actually loading any data.

!!! note
    You also cannot currently index into a `ManifestArray`, as arbitrary indexing would require loading data values to create the new array.
    We could imagine supporting indexing without loading data when slicing only along chunk boundaries, but this has not yet been implemented (see [GH issue #51](https://github.com/zarr-developers/VirtualiZarr/issues/51)).

## Zarr Groups

The full Zarr model (for a single group) includes multiple arrays, array names, named dimensions, group-level metadata and metadata on each array.
Whilst the a single duck-typed `ManifestArray` cannot store all of this information, a dictionary containing one or more `ManifestArrays` plus something to store the group-level metadata can.
VirtualiZarr has two different ways of doing this internally, which are used for different purposes.

## `ManifestGroup` and `ManifestStore` classes

A `ManifestGroup` is a dedicated class that contains multiple `ManifestArray`, plus group-level metadata.
It is designed to act similar to a Zarr group, such that a named collection of one or more `ManifestGroup` objects can be combined together to form a `ManifestStore`.

The `ManifestStore` (and `ManifestGroup`) classes are only used during `open_virtual_dataset`, to simplify the creation of virtual references and loading of variables from archival file formats.
You should therefore probably only use `ManifestStore` or `ManifestGroup` directly if you're planning to [write your own custom parser](custom_parsers.md) for an unsupported archival file format.

## "Virtual" Xarray Datasets

An alternate way to represent the contents of an entire Zarr group is to use an `xarray.Dataset` as the container of one or more `ManifestArray` objects.

This is what the virtual datasets we created in the usage guide represent - all the information in one entire Zarr group, but held as references to on-disk chunks instead of as in-memory arrays.
Any `ManifestGroup` (or single-group `ManifestStore`) can be converted to a virtual dataset.

The reason for having this alternate representation is that then problem of combining many archival files into one virtual Zarr store therefore becomes just a matter of opening each file using `open_virtual_dataset` and using [xarray's various combining functions](https://docs.xarray.dev/en/stable/user-guide/combining.html) to combine them into one aggregate virtual dataset.
See the [usage guide on combining virtual datasets](usage.md#combining-virtual-datasets) for more information.

!!! note
    In theory we could then invert the mapping to convert the virtual xarray Dataset back to a `ManifestStore` before persisting to the Icechunk/Kerchunk formats, but we don't currently do that, mainly because it makes handling loaded variables more complex.
