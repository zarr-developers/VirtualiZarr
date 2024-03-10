# VirtualiZarr

 **VirtualiZarr creates virtual Zarr stores for cloud-friendly access to archival data.**

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

### What's the difference between VirtualiZarr and Kerchunk?

The Kerchunk idea solves an incredibly important problem: accessing big archival datasets via a cloud-optimized pattern, but without copying or modifying the original data in any way. This is a win-win-win for users, data engineers, and data providers. Users see fast-opening zarr-compliant stores that work performantly with libraries like xarray and dask, data engineers can provide this speed by adding a lightweight virtualization layer on top of existing data - without having to ask anyone's permission, and data providers don't have to change anything about their legacy files for them to be used in a cloud-optimized way.

However, kerchunk's current design is limited:
- Store-level abstractions make combining datasets complicated, idiosyncratic, and requires duplicating logic that already exists in libraries like xarray,
- The kerchunk format for storing on-disk references requires the caller to understand it, usually via [`fsspec`](https://github.com/fsspec/filesystem_spec) (which is only implemented in python).

VirtualiZarr aims to build on the excellent ideas of kerchunk whilst solving the above problems:
- Using array-level abstractions instead is more modular, easier to reason about, allows convenient wrapping by high-level tools like xarray, and is simpler to parallelize,
- Writing the virtualized arrays out as a valid Zarr store directly (through new Zarr Extensions) will allow for Zarr implementations in any language to read the archival data.

### Installation

Currently you need to clone VirtualiZarr and install it locally:
```shell
git clone virtualizarr
pip install -e .
```

### Usage

Let's say you have a bunch of legacy files (e.g. netCDF) which together tile to form a large dataset. Let's imagine you already know how to use xarray to open these files and combine the opened dataset objects into one complete dataset. (If you don't then read the [xarray docs page on combining data](https://docs.xarray.dev/en/stable/user-guide/combining.html).)

```python
ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='h5netcdf',
    combine='by_coords',  # 'by_coords' requires reading coord data to determine concatenation order
)
```

You don't want to run this set of xarray operations every time you open this dataset as running commands like `open_mfdataset` can be expensive. Instead you would prefer to just be able to open a virtualized Zarr store (i.e. `xr.open_dataset('my_virtual_store.zarr')`), as that would open instantly, but still give access to the same data underneath.

**`VirtualiZarr` allows you to use the same xarray incantation you would normally use to open and combine all your files, but cache that result as a Zarr store.**

What's being cached here, you ask? We're effectively caching the result of performing all the various consistency checks that xarray performs when it combines newly-encountered datasets together. Once you have the new virtual Zarr store xarray is able to assume that this checking has already been done, and trusts your Zarr store enough to just open it instantly.

It looks like this:

```python
ds = xr.open_mfdataset(
    '/my/files*.nc',
    engine='virtualizarr',  # virtualizarr registers an xarray IO backend that returns ManifestArray objects
    combine='by_coords',  # 'by_coords' stills requires actually reading coordinate data
)

ds  # now wraps a bunch of virtual ManifestArray objects directly

ds.virtualize.to_zarr(store='out.zarr', spec='kerchunk')  # cache the combined dataset pattern to a new zarr store, in this case using the existing kerchunk specification
```

Now you can open your shiny new Zarr store instantly:

```python
fs = fsspec.filesystem('reference', fo='out.zarr')
m = fs.get_mapper('')

ds = xr.open_dataset(m, engine='kerchunk')
```

Since we serialized the cached results using the kerchunk specification then opening this zarr store still requires using fsspec via the kerchunk xarray backend.

### How does this work?

I'm glad you asked! We can think of the problem of providing virtualized zarr-like access to a set of legacy files in some other format as a series of steps:

1) **Read byte ranges** - We use the `kerchunk.backends` module to determine which byte ranges within a given legacy file would have to be read in order to get a specific chunk of data we want.
2) **Construct a representation of a single file (or array within a file)** - Kerchunk's backends return a nested dictionary representing an entire file, but we instead immediately parse this dict and wrap it up into a set of `ManifestArray` objects. The record of where to look to find the file and the byte ranges is stored in the `ManifestArray.manifest` attribute, which stores a `ChunkManifest` object. Both steps (1) and (2) are handled by the `'virtualizarr'` xarray backend, which returns one `xarray.Dataset` object per file, each wrapping multiple `ManifestArray` instances (as opposed to e.g. dask arrays).
3) **Deduce the concatenation order** - The desired order of concatenation can either be inferred from the order in which the datasets are supplied (which is what `xr.combined_nested` assumes), or it can be read from the coordinate data in the files (which is what `xr.combine_by_coords` does). If the ordering information is not present as a coordinate (e.g. because it's in the filename), a pre-processing step might be required.
4) **Perform checks that the desired concatenation is valid** - Whether called explicitly by the user or implicitly via `xr.combine_nested/combine_by_coords/open_mfdataset`, `xr.concat` is used to concatenate/stack the wrapped `ManifestArray` objects. When doing this xarray will spend time checking that the array objects and any coordinate indexes can be safely aligned and concatenated. Along with opening files, and loading coordinates in step (3), this is the main reason why `xr.open_mfdataset` can take a long time to return a dataset created from a large number of files.
5) **Combine into one big dataset** - `xr.concat` dispatches to the `concat/stack` methods of the underlying `ManifestArray` objects. These perform concatenation by merging their respective Chunk Manifests. Using xarray's `combine_*` methods means that we can handle multi-dimensional concatenations as well as merging many different variables.
6) **Serialize the result of the combining to disk** - The resultant `xr.Dataset` object contains `ManifestArray`s which contain the complete list of byte ranges for every chunk we might want to read. We now serialize this list out to disk, either using the [kerchunk specification](https://fsspec.github.io/kerchunk/spec.html#version-1), or in future we plan to use new Zarr extensions to write valid Zarr stores directly.
7) **Open the virtualized dataset from disk** - The virtualized zarr store can now be read from disk, skipping all the work we did above. Chunk reads from this store will be redirected to read the corresponding bytes in the legacy files.

**Note:** Using the `kerchunk` library alone will perform a similar set of steps overall, but because (3), (4), (5), and (6) are all performed by the `kerchunk.combine.MultiZarrToZarr` function, and no internal abstractions are exposed, the design is much less modular, and the use cases are limited by the kerchunk API surface.

### Development Status and Roadmap

VirtualiZarr is ready to use for many of the tasks that we are used to using kerchunk for, but the most general and powerful vision of this library can only be implemented once certain changes upstream in Zarr have occurred.

VirtualiZarr is therefore evolving in tandem with developments in the Zarr Specification, which then need to be implemented in specific Zarr reader implementations (especially the Zarr-Python V3 implementation). There is an [overall roadmap for this integration with Zarr](https://hackmd.io/t9Myqt0HR7O0nq6wiHWCDA), whose final completion requires acceptance of at least two new Zarr Enhancement Proposals (the ["Chunk Manifest"](https://github.com/zarr-developers/zarr-specs/issues/287) and ["Virtual Concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288) ZEPs).

Whilst we wait for these upstream changes, in the meantime VirtualiZarr aims to provide utility in a significant subset of cases, for example by enabling writing virtualized zarr stores out to the existing kerchunk references format, so that they can be read by fsspec today.

### Licence

Apache 2.0
