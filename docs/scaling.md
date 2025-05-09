(scaling)=

# Scaling

This page explains how to scale up your usage of VirtualiZarr to cloud-optimize large number of files.

## Pre-requisites

Before you attempt to use VirtualiZarr on a large number of files at once, you should check that you can successfully use the library on a small subset of your data.

In particular, you should check that:
- You can call `open_virtual_dataset` on one of your files.
- After doing this on a few files making up a representative subset of your data, you can concatenate them into one logical datacube without errors.
- You can serialize those virtual references to some format (e.g. Kerchunk/Icechunk) and read the data back.
- The data you read back is exactly what you would have expected to get if you read the data from the original files.

If you don't do these checks now, you might find that you deploy a large amount of resources to run VirtualiZarr on many files, only to hit a problem that you could have found much earlier.

## The need for parallelization

- Map-reduce

## Manual parallelism

- `open_virtual_dataset`

## The `parallel` kwarg to `open_virtual_mfdataset`

## Executors

If you prefer to do manual parallelism you can import and use these executors directly from `virtualizarr.parallel`.

### Serial

### Threads

### Dask

### Lithops

### Other

## Tips

### Caching

### Batching

### Retries
