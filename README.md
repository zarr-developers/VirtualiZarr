# VirtualiZarr

[![CI](https://github.com/zarr-developers/VirtualiZarr/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/zarr-developers/VirtualiZarr/actions?query=workflow%3ACI)
[![Code coverage](https://codecov.io/gh/zarr-developers/VirtualiZarr/branch/main/graph/badge.svg?flag=unittests)](https://codecov.io/gh/zarr-developers/VirtualiZarr)
[![Docs](https://readthedocs.org/projects/virtualizarr/badge/?version=latest)](https://virtualizarr.readthedocs.io/en/latest/)
[![Linted and Formatted with Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit Enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202-cb2533.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/zarr-developers/VirtualiZarr/main/pyproject.toml&logo=Python&logoColor=gold&label=Python)](https://docs.python.org)
[![slack](https://img.shields.io/badge/slack-virtualizarr-purple.svg?logo=slack)](https://join.slack.com/t/earthmover-community/shared_invite/zt-32to7398i-HorUXmzPzyy9U87yLxweIA)
[![Latest Release](https://img.shields.io/github/v/release/zarr-developers/VirtualiZarr)](https://github.com/zarr-developers/VirtualiZarr/releases)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/virtualizarr?label=pypi%7Cdownloads)](https://pypistats.org/packages/virtualizarr)
[![Conda - Downloads](https://img.shields.io/conda/d/conda-forge/virtualizarr
)](https://anaconda.org/conda-forge/virtualizarr)



## Cloud-Optimize your Scientific Data as a Virtual Zarr Datacube, using Xarray syntax.

The best way to distribute large scientific datasets is via the Cloud, in [Cloud-Optimized formats](https://guide.cloudnativegeo.org/) [^1]. But often this data is stuck in archival pre-Cloud file formats such as netCDF.

**VirtualiZarr[^2] makes it easy to create "Virtual" Zarr datacubes, allowing performant access to archival data as if it were in the Cloud-Optimized [Zarr format](https://zarr.dev/), _without duplicating any data_.**

Please see the [documentation](https://virtualizarr.readthedocs.io/en/stable/index.html).

### Features

* Create virtual references pointing to bytes inside an archival file with [`open_virtual_dataset`](https://virtualizarr.readthedocs.io/en/latest/usage.html#opening-files-as-virtual-datasets).
* Supports a [range of archival file formats](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare), including netCDF4 and HDF5, and has a pluggable system for supporting new formats.
* Access data via the zarr-python API by reading from the zarr-compatible [`ManifestStore`](https://virtualizarr.readthedocs.io/en/latest/generated/virtualizarr.manifests.ManifestStore.html).
* [Combine data from multiple files](https://virtualizarr.readthedocs.io/en/latest/usage.html#combining-virtual-datasets) into one larger datacube using [xarray's combining functions](https://docs.xarray.dev/en/stable/user-guide/combining.html), such as [`xarray.concat`](https://docs.xarray.dev/en/stable/generated/xarray.concat.html).
* Commit the virtual references to storage either using the [Kerchunk references](https://fsspec.github.io/kerchunk/spec.html) specification or the [Icechunk](https://icechunk.io/) transactional storage engine.
* Users access the virtual datacube simply as a single zarr-compatible store using [`xarray.open_zarr`](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html).

### Inspired by Kerchunk

VirtualiZarr grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [Kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk but in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

### Development Status and Roadmap

VirtualiZarr version 1 (mostly) achieved [feature parity](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) with kerchunk's logic for combining datasets, providing an easier way to manipulate kerchunk references in memory and generate kerchunk reference files on disk.

VirtualiZarr version 2 brings:

- Zarr v3 support
- A pluggable system of "parsers" for virtualizing custom file formats
- The `ManifestStore` abstraction, which allows for loading data without serializing to Kerchunk/Icechunk first
- Integration with [`obstore`](https://developmentseed.org/obstore/latest/)
- Reference parsing that doesn't rely on kerchunk under the hood
- The ability to use "parsers" to load data directly from archival file formats into Zarr and/or Xarray

Future VirtualiZarr development will focus on generalizing and upstreaming useful concepts into the Zarr specification, the Zarr-Python library, Xarray, and possibly some new packages.

We have a lot of ideas, including:
- [Zarr-native on-disk chunk manifest format](https://github.com/zarr-developers/zarr-specs/issues/287)
- ["Virtual concatenation"](https://github.com/zarr-developers/zarr-specs/issues/288) of separate Zarr arrays
- ManifestArrays as an [intermediate layer in-memory](https://github.com/zarr-developers/VirtualiZarr/issues/71) in Zarr-Python
- [Separating CF-related Codecs from xarray](https://github.com/zarr-developers/VirtualiZarr/issues/68#issuecomment-2197682388)

If you see other opportunities then we would love to hear your ideas!

### Talks and Presentations

- 2025/04/30 - Cloud-Native Geospatial Forum - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-and-icechunk-build-a-cloud-optimized-datacube-in-3-lines) / [Recording](https://youtu.be/QBkZQ53vE6o)
- 2024/11/21 - MET Office Architecture Guild - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-talk-at-met-office)
- 2024/11/13 - Cloud-Native Geospatial conference - Raphael Hagen - [Slides](https://decks.carbonplan.org/cloud-native-geo/11-13-24)
- 2024/07/24 - ESIP Meeting - Sean Harkins - [Event](https://2024julyesipmeeting.sched.com/event/1eVP6) / [Recording](https://youtu.be/T6QAwJIwI3Q?t=3689)
- 2024/05/15 - Pangeo showcase - Tom Nicholas - [Event](https://discourse.pangeo.io/t/pangeo-showcase-virtualizarr-create-virtual-zarr-stores-using-xarray-syntax/4127/2) / [Recording](https://youtu.be/ioxgzhDaYiE) / [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-create-virtual-zarr-stores-using-xarray-syntax)

### Credits

This package was originally developed by [Tom Nicholas](https://github.com/TomNicholas) whilst working at [[C]Worthy](https://cworthy.org), who deserve credit for allowing him to prioritise a generalizable open-source solution to the dataset virtualization problem. VirtualiZarr is now a community-owned multi-stakeholder project.

### Licence

Apache 2.0

### References

[^1]: [_Cloud-Native Repositories for Big Scientific Data_, Abernathey et. al., _Computing in Science & Engineering_.](https://ieeexplore.ieee.org/abstract/document/9354557)

[^2]: (Pronounced "Virtual-Eye-Zarr" - like "virtualizer" but more piratey 🦜)
