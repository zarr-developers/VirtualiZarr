(usage)=
# Usage

This page explains how to use VirtualiZarr today, by introducing the key concepts one-by-one.

## Opening files as virtual datasets

VirtualiZarr is for manipulating "virtual" references to pre-existing data stored on disk in a variety of formats, by representing it in terms of the [Zarr data model](https://zarr-specs.readthedocs.io/en/latest/specs.html) of chunked N-dimensional arrays.

If we have a pre-existing netCDF file on disk,

```python
import xarray as xr

# create an example pre-existing netCDF4 file
ds = xr.tutorial.open_dataset('air_temperature')
ds.to_netcdf('air.nc')
```

We can open a virtual representation of this file using {py:func}`open_virtual_dataset <virtualizarr.xarray.open_virtual_dataset>`.

```python
from virtualizarr import open_virtual_dataset

vds = open_virtual_dataset('air.nc')
```

(Notice we did not have to explicitly indicate the file format, as {py:func}`open_virtual_dataset <virtualizarr.xarray.open_virtual_dataset>` will attempt to automatically infer it.)

```{note}
In future we would like for it to be possible to just use `xr.open_dataset`, e.g.

    import virtualizarr

    vds = xr.open_dataset('air.nc', engine='virtualizarr')

but this requires some [upstream changes](https://github.com/TomNicholas/VirtualiZarr/issues/35) in xarray.
```

Printing this "virtual dataset" shows that although it is an instance of `xarray.Dataset`, unlike a typical xarray dataset, it does not contain numpy or dask arrays, but instead it wraps {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects.

```python
vds
```
```
<xarray.Dataset> Size: 8MB
Dimensions:  (time: 2920, lat: 25, lon: 53)
Coordinates:
    lat      (lat) float32 100B ManifestArray<shape=(25,), dtype=float32, chu...
    lon      (lon) float32 212B ManifestArray<shape=(53,), dtype=float32, chu...
    time     (time) float32 12kB ManifestArray<shape=(2920,), dtype=float32, ...
Data variables:
    air      (time, lat, lon) int16 8MB ManifestArray<shape=(2920, 25, 53), d...
Attributes:
    Conventions:  COARDS
    description:  Data is from NMC initialized reanalysis\n(4x/day).  These a...
    platform:     Model
    references:   http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanaly...
    title:        4x daily NMC reanalysis (1948)
```

These {py:class}`ManifestArray <virtualizarr.manifests.ManifestArray>` objects are each a virtual reference to some data in the `air.nc` netCDF file, with the references stored in the form of "Chunk Manifests".

## Chunk Manifests

TODO: Basic concept of what a chunk manifest is

## `ManifestArray` class

TODO: A `ManifestArray` as an array-like wrapper of a single chunk manifest, and why that's like a virtualized zarr array

## Virtual Xarray Datasets as Zarr Groups

TODO: How the whole xarray dataset maps to the zarr model

## Concatenation via xarray using given order (i.e. without indexes)

TODO: How concatenating in given order works

TODO: Note on how this will only work if you have the correct fork of xarray

TODO: Note on how this could be done using `open_mfdataset(..., combine='nested')` in future

## Concatenation via xarray using order inferred from indexes

TODO: How to concatenate with order inferred from indexes automatically

TODO: Note on how this could be done using `open_mfdataset(..., combine='by_coords')` in future

## Writing virtual stores to disk

### Writing as kerchunk format and reading via fsspec

TODO: Explanation of how this uses kerchunks format

TODO: Reading using fsspec

### Writing as Zarr

TODO: Explanation of how this requires changes in zarr upstream to be able to read it
