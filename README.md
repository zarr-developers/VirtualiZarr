# VirtualiZarr

[![CI](https://github.com/zarr-developers/VirtualiZarr/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/zarr-developers/VirtualiZarr/actions?query=workflow%3ACI)
[![Code coverage](https://codecov.io/gh/zarr-developers/VirtualiZarr/branch/main/graph/badge.svg?flag=unittests)](https://codecov.io/gh/zarr-developers/VirtualiZarr)
[![Docs](https://readthedocs.org/projects/virtualizarr/badge/?version=latest)](https://virtualizarr.readthedocs.io/en/latest/)
[![Linted and Formatted with Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![pre-commit Enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202-cb2533.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/zarr-developers/VirtualiZarr/main/pyproject.toml&logo=Python&logoColor=gold&label=Python)](https://docs.python.org)

[![Latest Release](https://img.shields.io/github/v/release/zarr-developers/VirtualiZarr)](https://github.com/zarr-developers/VirtualiZarr/releases)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/virtualizarr?label=pypi%7Cdownloads)](https://pypistats.org/packages/virtualizarr)
[![Conda - Downloads](https://img.shields.io/conda/d/conda-forge/virtualizarr
)](https://anaconda.org/conda-forge/virtualizarr)

**VirtualiZarr creates virtual Zarr stores for cloud-friendly access to archival data, using familiar xarray syntax.**

VirtualiZarr (pronounced like "virtualizer" but more piratey) grew out of [discussions](https://github.com/fsspec/kerchunk/issues/377) on the [kerchunk repository](https://github.com/fsspec/kerchunk), and is an attempt to provide the game-changing power of kerchunk in a zarr-native way, and with a familiar array-like API.

You now have a choice between using VirtualiZarr and Kerchunk: VirtualiZarr provides [almost all the same features](https://virtualizarr.readthedocs.io/en/latest/faq.html#how-do-virtualizarr-and-kerchunk-compare) as Kerchunk.

_Please see the [documentation](https://virtualizarr.readthedocs.io/en/stable/index.html)_

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

### Presentations

- 2024/11/21 - MET Office Architecture Guild - Tom Nicholas - [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-talk-at-met-office)
- 2024/11/13 - Cloud-Native Geospatial conference - Raphael Hagen - [Slides](https://decks.carbonplan.org/cloud-native-geo/11-13-24)
- 2024/07/24 - ESIP Meeting - Sean Harkins - [Event](https://2024julyesipmeeting.sched.com/event/1eVP6) / [Recording](https://youtu.be/T6QAwJIwI3Q?t=3689)
- 2024/05/15 - Pangeo showcase - Tom Nicholas - [Event](https://discourse.pangeo.io/t/pangeo-showcase-virtualizarr-create-virtual-zarr-stores-using-xarray-syntax/4127/2) / [Recording](https://youtu.be/ioxgzhDaYiE) / [Slides](https://speakerdeck.com/tomnicholas/virtualizarr-create-virtual-zarr-stores-using-xarray-syntax)

### Credits

This package was originally developed by [Tom Nicholas](https://github.com/TomNicholas) whilst working at [[C]Worthy](cworthy.org), who deserve credit for allowing him to prioritise a generalizable open-source solution to the dataset virtualization problem. VirtualiZarr is now a community-owned multi-stakeholder project.

### Licence

Apache 2.0
