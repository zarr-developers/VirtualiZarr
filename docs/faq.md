# FAQ

## Usage questions

### I'm an Xarray user but unfamiliar with Zarr/Cloud - might I still want this?

Potentially yes.

Let's say you have a bunch of archival files (e.g. netCDF) which together tile along one or more dimensions to form a large dataset.
Let's also imagine you already know how to use xarray to open these files and combine the opened dataset objects into one complete dataset.
(If you don't then read the [xarray docs page on combining data](https://docs.xarray.dev/en/stable/user-guide/combining.html).)

```python
# open_mfdataset does a lot of checks, so can take a while
ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='h5netcdf',
    combine='nested',
)
ds  # the complete lazy xarray dataset
```

However, you don't want to run this set of xarray operations every time you open this dataset, as running commands like `xr.open_mfdataset` can be expensive.
Instead you would prefer to just be able to open a single pre-saved virtual store that points to all your data, as that would open instantly (using `xr.open_dataset('my_virtual_store.zarr')`), but still give access to the same data underneath.

**`VirtualiZarr` aims to allow you to use the same xarray incantation you would normally use to open and combine all your files, but cache that result as a virtual Zarr store.**

You can think of this as effectively caching the result of performing all the various consistency checks that xarray performs when it combines newly-encountered datasets together.
Once you have the new virtual Zarr store xarray is able to assume that this checking has already been done, and trusts your Zarr store enough to just open it instantly.

```{note}
This means you should not change or add to any of the files comprising the store once created. If you want to make changes or add new data, you should look into using [Icechunk](https://icechunk.io/) instead.
```

As Zarr can read data that lives on filesystems too, this can be useful even if you don't plan to put your data in the cloud.
You can create the virtual store once (e.g. as soon as your HPC simulation finishes) and then opening that dataset will be much faster than using `open_mfdataset` each time.

### Is this compatible with Icechunk?

Very much so! VirtualiZarr allows you to ingest data as virtual references and write those references into an [Icechunk](https://icechunk.io/) Store. See the [Icechunk documentation on creating virtual datasets](https://icechunk.io/icechunk-python/virtual/#creating-a-virtual-dataset-with-virtualizarr).

In general once the Icechunk specification reaches a stable v1.0, we would recommend using that over Kerchunk's references format, in order to take advantage of transactional updates, version controlled history, and faster access speeds.

### I have already Kerchunked my data, do I have to redo that?

No - you can simply open the Kerchunk-formatted references you already have into VirtualiZarr directly. Then you can manipulate them, or re-save them into a new format, such as  [Icechunk](https://icechunk.io/):

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset('refs.json')
# vds = open_virtual_dataset('refs.parq')  # kerchunk parquet files are supported too

vds.virtualize.to_icechunk(icechunkstore)
```

### I already have some data in Zarr, do I have to resave it?

No! VirtualiZarr can (well, [soon will be able to](https://github.com/zarr-developers/VirtualiZarr/issues/262)) create virtual references pointing to existing Zarr stores in the same way as for other file formats.

### Can I add a new reader for my custom file format?

There are a lot of archival file formats which could potentially be represented as virtual zarr references (see [this issue](https://github.com/zarr-developers/VirtualiZarr/issues/218) listing some examples).
VirtualiZarr ships with some readers for common formats (e.g. netCDF/HDF5), but you may want to write your own reader for some other file format.

VirtualiZarr is designed in a way to make this as straightforward as possible.
If you want to do this then [this comment](https://github.com/zarr-developers/VirtualiZarr/issues/262#issuecomment-2429968244
) will be helpful.

You can also use this approach to write a reader that starts from a kerchunk-formatted virtual references dict.

Currently if you want to call your new reader from `virtualizarr.open_virtual_dataset` you would need to open a PR to this repository, but we plan to generalize this system to allow 3rd party libraries to plug in via an entrypoint (see [issue #245](https://github.com/zarr-developers/VirtualiZarr/issues/245)).

## How does this actually work?

I'm glad you asked! We can think of the problem of providing virtualized zarr-like access to a set of archival files in some other format as a series of steps:

1) **Read byte ranges** - We use various [virtualizarr readers](https://github.com/zarr-developers/VirtualiZarr/tree/main/virtualizarr/readers) to determine which byte ranges within a given archival file would have to be read in order to get a specific chunk of data we want. Several of these readers work by calling one of the [kerchunk file format backends](https://fsspec.github.io/kerchunk/reference.html#file-format-backends) and parsing the output.
2) **Construct a representation of a single file (or array within a file)** - Kerchunk's backends return a nested dictionary representing an entire file, but we instead immediately parse this dict and wrap it up into a set of `ManifestArray` objects. The record of where to look to find the file and the byte ranges is stored under the `ManifestArray.manifest` attribute, in a `ChunkManifest` object. Both steps (1) and (2) are handled by the `virtualizarr.open_virtual_dataset`, which returns one `xarray.Dataset` object for the given file, which wraps multiple `ManifestArray` instances (as opposed to e.g. numpy/dask arrays).
3) **Deduce the concatenation order** - The desired order of concatenation can either be inferred from the order in which the datasets are supplied (which is what `xr.combined_nested` assumes), or it can be read from the coordinate data in the files (which is what `xr.combine_by_coords` does). If the ordering information is not present as a coordinate (e.g. because it's in the filename), a pre-processing step might be required.
4) **Check that the desired concatenation is valid** - Whether called explicitly by the user or implicitly via `xr.combine_nested/combine_by_coords/open_mfdataset`, `xr.concat` is used to concatenate/stack the wrapped `ManifestArray` objects. When doing this xarray will spend time checking that the array objects and any coordinate indexes can be safely aligned and concatenated. Along with opening files, and loading coordinates in step (3), this is the main reason why `xr.open_mfdataset` can take a long time to return a dataset created from a large number of files.
5) **Combine into one big dataset** - `xr.concat` dispatches to the `concat/stack` methods of the underlying `ManifestArray` objects. These perform concatenation by merging their respective Chunk Manifests. Using xarray's `combine_*` methods means that we can handle multi-dimensional concatenations as well as merging many different variables.
6) **Serialize the combined result to disk** - The resultant `xr.Dataset` object wraps `ManifestArray` objects which contain the complete list of byte ranges for every chunk we might want to read. We now serialize this information to disk, either using the [Kerchunk specification](https://fsspec.github.io/kerchunk/spec.html#version-1), or the [Icechunk specification](https://icechunk.io/spec/).
7) **Open the virtualized dataset from disk** - The virtualized zarr store can now be read from disk, avoiding redoing all the work we did above and instead just opening all the virtualized data immediately. Chunk reads will be redirected to read the corresponding bytes in the original archival files.

The above steps could also be performed using the `kerchunk` library alone, but because (3), (4), (5), and (6) are all performed by the `kerchunk.combine.MultiZarrToZarr` function, and no internal abstractions are exposed, kerchunk's design is much less modular, and the use cases are limited by kerchunk's API surface.

## How do VirtualiZarr and Kerchunk compare?

You have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides almost all the same features as Kerchunk.

Users of Kerchunk may find the following comparison table useful, which shows which features of Kerchunk map on to which features of VirtualiZarr.

| Component / Feature                                                      | Kerchunk                                                                                                                            | VirtualiZarr                                                                                                                                     |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Generation of references from archival files (1)**                     |                                                                                                                                     |                                                                                                                                                  |
| From a netCDF4/HDF5 file                                                 | `kerchunk.hdf.SingleHdf5ToZarr`                                                                                                     | `open_virtual_dataset(..., filetype='hdf5')`, via `kerchunk.hdf.SingleHdf5ToZarr`                                                            |
| From a netCDF3 file                                                      | `kerchunk.netCDF3.NetCDF3ToZarr`                                                                                                    | `open_virtual_dataset(..., filetype='netcdf3')`, via `kerchunk.netCDF3.NetCDF3ToZarr`                                                                                     |
| From a COG / tiff file                                                   | `kerchunk.tiff.tiff_to_zarr`                                                                                                        | `open_virtual_dataset(..., filetype='tiff')`, via `kerchunk.tiff.tiff_to_zarr` or potentially `tifffile` (❌ Not yet implemented - see [issue #291](https://github.com/zarr-developers/VirtualiZarr/issues/291))                                                               |
| From a Zarr v2 store                                                     | `kerchunk.zarr.ZarrToZarr`                                                                                                          | `open_virtual_dataset(..., filetype='zarr')` (❌ Not yet implemented - see [issue #262](https://github.com/zarr-developers/VirtualiZarr/issues/262))                                                                                       |
| From a Zarr v3 store                                                     | ❌                                                                                                          | `open_virtual_dataset(..., filetype='zarr')` (❌ Not yet implemented - see [issue #262](https://github.com/zarr-developers/VirtualiZarr/issues/262))                                                                                       |
| From a GRIB2 file                                                        | `kerchunk.grib2.scan_grib`                                                                                                          | `open_virtual_datatree(..., filetype='grib')` (❌ Not yet implemented - see [issue #11](https://github.com/zarr-developers/VirtualiZarr/issues/11))                                                                                |
| From a FITS file                                                         | `kerchunk.fits.process_file`                                                                                                        | `open_virtual_dataset(..., filetype='fits')`, via `kerchunk.fits.process_file`                                                                                      |
| From a HDF4 file                                                         | `kerchunk.hdf4.HDF4ToZarr`                                                                                                        | `open_virtual_dataset(..., filetype='hdf4')`, via `kerchunk.hdf4.HDF4ToZarr` (❌ Not yet implemented - see [issue #216](https://github.com/zarr-developers/VirtualiZarr/issues/216))                                                        |
| From a [DMR++](https://opendap.github.io/DMRpp-wiki/DMRpp.html) metadata file                                                    | ❌                                                                                                        | `open_virtual_dataset(..., filetype='dmrpp')`, via `virtualizarr.readers.dmrpp.DMRParser`                                                                                      |
| From existing kerchunk JSON/parquet references                                                 | `kerchunk.combine.MultiZarrToZarr(append=True)`                                                                                                       | `open_virtual_dataset(..., filetype='kerchunk')`                                                                                      |
| **In-memory representation (2)**                                         |                                                                                                                                     |                                                                                                                                                  |
| In-memory representation of byte ranges for single array                 | Part of a "reference `dict`" with keys for each chunk in array                                                                      | `ManifestArray` instance (wrapping a `ChunkManifest` instance)                                                                                   |
| In-memory representation of actual data values                           | Encoded bytes directly serialized into the "reference `dict`", created on a per-chunk basis using the `inline_threshold` kwarg      | `numpy.ndarray` instances, created on a per-variable basis using the `loadable_variables` kwarg                                                  |
| In-memory representation of entire file / store                          | Nested "reference `dict`" with keys for each array in file                                                                          | `xarray.Dataset` with variables wrapping `ManifestArray` instances (or `numpy.ndarray` instances)                                                |
| **Manipulation of in-memory references (3, 4 & 5)**                      |                                                                                                                                     |                                                                                                                                                  |
| Combining references to multiple arrays representing different variables | `kerchunk.combine.MultiZarrToZarr`                                                                                                  | `xarray.merge`                                                                                                                                   |
| Combining references to multiple arrays representing the same variable   | `kerchunk.combine.MultiZarrToZarr` using the `concat_dims` kwarg                                                                    | `xarray.concat`                                                                                                                                  |
| Combining references in coordinate order                                 | `kerchunk.combine.MultiZarrToZarr` using the `coo_map` kwarg                                                                        | `xarray.combine_by_coords` with in-memory coordinate variables loaded via the `loadable_variables` kwarg                            |
| Combining along multiple dimensions without coordinate data              | ❌                                                                                                                                 | `xarray.combine_nested`                                                                                                                          |
| Dropping variables              | `kerchunk.combine.drop`                                                                                                                                 | `xarray.Dataset.drop_vars`, or `open_virtual_dataset(..., drop_variables=...)`                                                                                                                          |
| Renaming variables              | ❌                                                                                                                                  | `xarray.Dataset.rename_vars`                                                                                                                          |
| Renaming dimensions              | ❌                                                                                                                                  | `xarray.Dataset.rename_dims`                                                                                                                          |
| Renaming manifest file paths | `kerchunk.utils.rename_target`                                                                                                                                  | `vds.virtualize.rename_paths`                                                                                                                          |
| Splitting uncompressed data into chunks | `kerchunk.utils.subchunk`                                                                                                                                  | `xarray.Dataset.chunk` (❌ Not yet implemented - see [PR #199](https://github.com/zarr-developers/VirtualiZarr/pull/199))
| Selecting specific chunks | ❌                                                                                                                                  | `xarray.Dataset.isel` (❌ Not yet implemented - see [issue #51](https://github.com/zarr-developers/VirtualiZarr/issues/51))                                                                                                                          |
**Parallelization**                                                      |                                                                                                                                     |                                                                                                                                                  |
| Parallelized generation of references                                    | Wrapping kerchunk's opener inside `dask.delayed`                                                                                    | Wrapping `open_virtual_dataset` inside `dask.delayed` (⚠️ Untested)
| Parallelized combining of references (tree-reduce)                       | `kerchunk.combine.auto_dask`                                                                                                        | Wrapping `ManifestArray` objects within `dask.array.Array` objects inside `xarray.Dataset` to use dask's `concatenate` (⚠️ Untested)                         |
| **On-disk serialization (6) and reading (7)**                            |                                                                                                                                     |                                                                                                                                                  |
| Kerchunk reference format as JSON                                        | `ujson.dumps(h5chunks.translate())` , then read using an `fsspec.filesystem` mapper                                | `ds.virtualize.to_kerchunk('combined.json', format='JSON')` , then read using an `fsspec.filesystem` mapper                                      |
| Kerchunk reference format as parquet                                     | `df.refs_to_dataframe(out_dict, "combined.parq")`, then read using an `fsspec` `ReferenceFileSystem` mapper | `ds.virtualize.to_kerchunk('combined.parq', format=parquet')` , then read using an `fsspec` `ReferenceFileSystem` mapper |
| Zarr v3 store with `manifest.json` files                                 | ❌                                                                                                                                 | `ds.virtualize.to_zarr()`, then read via any Zarr v3 reader which implements the manifest storage transformer ZEP                                |
| [Icechunk](https://icechunk.io/) store                          | ❌                                                                                                                                 | `ds.virtualize.to_icechunk()`, then read back via xarray (requires zarr-python v3).                                |

## Development

### Why a new project?

The reasons why VirtualiZarr has been developed as separate project rather than by contributing to the Kerchunk library upstream are:
- Kerchunk aims to support non-Zarr-like formats too [(1)](https://github.com/fsspec/kerchunk/issues/386#issuecomment-1795379571) [(2)](https://github.com/zarr-developers/zarr-specs/issues/287#issuecomment-1944439368), whereas VirtualiZarr is more strictly scoped, and may eventually be very tighted integrated with the Zarr-Python library itself.
- Whilst some features of VirtualiZarr currently require importing Kerchunk, Kerchunk is an optional dependency, and the VirtualiZarr roadmap aims to at some point not share any code with the Kerchunk library, nor ever require importing it. (You would nevertheless still be able to write out references in the Kerchunk format though!)
- The API design of VirtualiZarr is deliberately [completely different](https://github.com/fsspec/kerchunk/issues/377#issuecomment-1922688615) to Kerchunk's API, so integration into Kerchunk would have meant duplicated functionality.
- Refactoring Kerchunk's existing API to maintain backwards compatibility would have been [challenging](https://github.com/fsspec/kerchunk/issues/434).

### What is the Development Status and Roadmap?

VirtualiZarr version 1 (mostly) achieves [feature parity](#how-do-virtualizarr-and-kerchunk-compare) with kerchunk's logic for combining datasets, providing an easier way to manipulate kerchunk references in memory and generate kerchunk reference files on disk.

Future VirtualiZarr development will focus on generalizing and upstreaming useful concepts into the Zarr specification, the Zarr-Python library, Xarray, and possibly some new packages.

We have a lot of ideas, including:
- [Zarr v3 support](https://github.com/zarr-developers/VirtualiZarr/issues/17)
- [Zarr-native on-disk chunk manifest format](https://github.com/zarr-developers/zarr-specs/issues/287)
- ["Virtual concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288) of separate Zarr arrays
- ManifestArrays as an [intermediate layer in-memory](https://github.com/zarr-developers/VirtualiZarr/issues/71) in Zarr-Python
- [Separating CF-related Codecs from xarray](https://github.com/zarr-developers/VirtualiZarr/issues/68#issuecomment-2197682388)
- [Generating references without kerchunk](https://github.com/zarr-developers/VirtualiZarr/issues/78)

If you see other opportunities then we would love to hear your ideas!
