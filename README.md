# VirtualiZarr

**VirtualiZarr creates virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.**

VirtualiZarr (pronounced "virtualize-arr") grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

_Please see the [documentation](https://virtualizarr.readthedocs.io/en/latest/)_

### Development Status and Roadmap

VirtualiZarr is ready to use for many of the tasks that we are used to using kerchunk for, but the most general and powerful vision of this library can only be implemented once certain changes upstream in Zarr have occurred.

VirtualiZarr is therefore evolving in tandem with developments in the Zarr Specification, which then need to be implemented in specific Zarr reader implementations (especially the Zarr-Python V3 implementation). There is an [overall roadmap for this integration with Zarr](https://hackmd.io/t9Myqt0HR7O0nq6wiHWCDA), whose final completion requires acceptance of at least two new Zarr Enhancement Proposals (the ["Chunk Manifest"](https://github.com/zarr-developers/zarr-specs/issues/287) and ["Virtual Concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288) ZEPs).

Whilst we wait for these upstream changes, in the meantime VirtualiZarr aims to provide utility in a significant subset of cases, for example by enabling writing virtualized zarr stores out to the existing kerchunk references format, so that they can be read by fsspec today.

### Licence

Apache 2.0
