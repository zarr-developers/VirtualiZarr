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

TODO

### Development Status and Roadmap

VirtualiZarr is evolving in tandem with developments in the Zarr Specification, which then need to be implemented in specific Zarr reader implementations (especially the Zarr-Python V3 implementation). There is an [overall roadmap for upstreaming Kerchunk's functionality into Zarr](https://hackmd.io/t9Myqt0HR7O0nq6wiHWCDA), whose final completion requires acceptance of at least two new Zarr Enhancement Proposals (["Chunk Manifest"](https://github.com/zarr-developers/zarr-specs/issues/287) and ["Virtual Concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288)).

Whilst we wait for these upstream changes, in the meantime VirtualiZarr aims to provide utility in a significant subset of cases, for example by enabling writing virtualized zarr stores out to the existing kerchunk references format, so that they can be read by fsspec today.
