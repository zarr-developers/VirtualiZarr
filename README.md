# VirtualiZarr

**VirtualiZarr creates virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.**

VirtualiZarr (pronounced like "virtualize" but more piratey) grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

_Please see the [documentation](https://virtualizarr.readthedocs.io/en/latest/)_

### Development Status and Roadmap

VirtualiZarr version 1 (mostly) achieves [feature parity](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) with kerchunk's logic for combining datasets, providing an easier way to manipulate kerchunk references in memory and generate kerchunk reference files on disk.

Future VirtualiZarr development will focus on generalizing and upstreaming useful concepts into the Zarr specification, the Zarr-Python library, Xarray, and possibly some new packages.

We have a lot of ideas, including:
- [Zarr v3 support](https://github.com/zarr-developers/VirtualiZarr/issues/17)
- [Zarr-native on-disk chunk manifest format](https://github.com/zarr-developers/zarr-specs/issues/287)
- ["Virtual concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288) of separate Zarr arrays
- ManifestArrays as an [intermediate layer in-memory](https://github.com/zarr-developers/VirtualiZarr/issues/71) in Zarr-Python
- [Separating CF-related Codecs from xarray](https://github.com/zarr-developers/VirtualiZarr/issues/68#issuecomment-2197682388)
- [Generating references without kerchunk](https://github.com/zarr-developers/VirtualiZarr/issues/78)

If you see other opportunities then we would love to hear your ideas!

### Credits

This package was originally developed by [Tom Nicholas](https://github.com/TomNicholas) whilst working at [[C]Worthy](cworthy.org), who deserve credit for allowing him to prioritise a generalizable open-source solution to the dataset virtualization problem. VirtualiZarr is now a community-owned multi-stakeholder project.

### Licence

Apache 2.0
