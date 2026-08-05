# End-to-end examples

The following examples demonstrate the use of VirtualiZarr to create virtual datasets of various kinds.

## V2 Examples

These examples use the VirtualiZarr 2.x API with `obstore` for cloud storage access:

1. [Virtualizing GOES-16 satellite data (basic)](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes_basic.py) - Virtualizes a GOES-16 file using a standard obstore. Use as a baseline for performance comparison.
2. [Virtualizing GOES-16 satellite data with optimized store wrappers](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes_with_caching_stores.py) - Demonstrates using `CachingReadableStore` and `SplittingReadableStore` from `obspec-utils` to optimize metadata access when virtualizing remote HDF5/NetCDF files.

### GOES-16 archive notebooks

This notebook accompanies the [_Old format, no problem!: Cloud-optimizing the GOES-16 archive as Virtual Zarr_](https://www.earthmover.io/blog/virtual-zarr) blog post, which virtualizes NOAA's entire 7-year GOES-16 archive into a single virtual Zarr datacube:

1. [Ingesting the GOES-16 archive](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes-16-ingest.ipynb) - End-to-end notebook building the virtual store from the netCDF archive.

### ITS_LIVE glacier-velocity mosaic

1. [Mosaicking ITS_LIVE granules into a virtual cube](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/its_live.ipynb) - Aligns granules that share a global grid but cover different regions using native xarray `concat(..., join="outer")`, stacks them along `time`, and writes the sparse virtual cube to Icechunk.

## V1 Examples

!!! note
    The V1 examples listed here use a pre-2.0 release of VirtualiZarr.

1. [Appending new daily NOAA SST data to Icechunk](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V1/append/noaa-cdr-sst.ipynb)
2. [Parallel reference generation using Coiled Functions](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V1/coiled/terraclimate.ipynb)
3. [Serverless parallel reference generation using Lithops](https://github.com/zarr-developers/VirtualiZarr/tree/main/examples/V1/virtualizarr-with-lithops)
4. [MUR SST Virtual and Zarr Icechunk Store Generation using Lithops](https://github.com/zarr-developers/VirtualiZarr/tree/main/examples/V1/mursst-icechunk-with-lithops)

## Next steps

Several of these examples virtualize large archives. To learn how to generate many virtual references in parallel, see [Scaling](scaling.md).
