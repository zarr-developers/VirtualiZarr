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

### Development Status and Roadmap

VirtualiZarr is ready to use on kerchunk-type problems right now, but the most general and powerful vision of this library can only be implemented once certain changes upstream in Zarr have occurred.

VirtualiZarr is therefore evolving in tandem with developments in the Zarr Specification, which then need to be implemented in specific Zarr reader implementations (especially the Zarr-Python V3 implementation). There is an [overall roadmap for upstreaming Kerchunk's functionality into Zarr](https://hackmd.io/t9Myqt0HR7O0nq6wiHWCDA), whose final completion requires acceptance of at least two new Zarr Enhancement Proposals (["Chunk Manifest"](https://github.com/zarr-developers/zarr-specs/issues/287) and ["Virtual Concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288)).

Whilst we wait for these upstream changes, in the meantime VirtualiZarr aims to provide utility in a significant subset of cases, for example by enabling writing virtualized zarr stores out to the existing kerchunk references format, so that they can be read by fsspec today.
