(usage)=
# Usage

This page explains how to use VirtualiZarr today, by introducing the key concepts one-by-one.

## Opening files as virtual datasets

TODO: `open_virtual_dataset`

TODO: Note about how in future this will be possible using only `xr.open_dataset`

## Chunk Manifests

TODO: Basic concept of what a chunk manifest is

## `ManifestArray` class

TODO: A `ManifestArray` as an array-like wrapper of a single chunk manifest, and why that's like a virtualized zarr array

## Virtual Xarray Datasets

TODO: How the whole xarray dataset maps to the zarr model

## Concatenation via xarray using given order (i.e. without indexes)

TODO: How concatenating in given order works

TODO: Note on how this will only work if you have the correct fork of xarray

TODO: Note on how this could be done using `open_mfdataset(..., combine='nested')` in future

## Concatenation via xarray using order inferred from indexes

TODO: How to concatenate with order inferred from indexes automatically

TODO: Note on how this could be done using `open_mfdataset(..., combine='by_coords')` in future

## Writing virtual store to disk

### Using kerchunk

TODO: Explanation of how this uses kerchunks format

TODO: Reading using fsspec

### Using zarr

TODO: Explanation of how this requires changes in zarr upstream to be able to read it
