
# Data structures

This page explains the data structures available as part of VirtualiZarr today, by introducing the key concepts one-by-one.

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
As a single Zarr array has only one array-level set of compression codecs by definition, concatenation of arrays from files saved to disk with differing codecs cannot be achieved through concatenation of `ManifestArray` objects. Implementing this feature will require a more abstract and general notion of concatenation, see [GH issue #5](https://github.com/zarr-developers/VirtualiZarr/issues/5).
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
You also cannot currently index into a `ManifestArray`, as arbitrary indexing would require loading data values to create the new array. We could imagine supporting indexing without loading data when slicing only along chunk boundaries, but this has not yet been implemented (see [GH issue #51](https://github.com/zarr-developers/VirtualiZarr/issues/51)).
```

## `ManifestGroup` group

The full Zarr model (for a single group) includes multiple arrays, array names, named dimensions, and arbitrary dictionary-like attrs on each array. Whilst the duck-typed `ManifestArray` cannot store all of this information, an `zarr.Group` containing multiple `ManifestArray`s maps neatly to the Zarr model. This is what the `ManifestGroup` represents - all the information in one entire Zarr group, but held as references to on-disk chunks instead of as in-memory arrays.
