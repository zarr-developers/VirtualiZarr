# End-to-end examples

The following examples demonstrate the use of VirtualiZarr to create virtual datasets of various kinds.

## V2 Examples

These examples use the VirtualiZarr 2.x API with `obstore` for cloud storage access:

1. [Virtualizing GOES-16 satellite data (basic)](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes_basic.py) - Virtualizes a GOES-16 file using a standard obstore. Use as a baseline for performance comparison.
2. [Virtualizing GOES-16 satellite data with optimized store wrappers](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V2/goes_with_caching_stores.py) - Demonstrates using `CachingReadableStore` and `SplittingReadableStore` from `obspec-utils` to optimize metadata access when virtualizing remote HDF5/NetCDF files.

## V1 Examples

!!! note
    The V1 examples listed here use a pre-2.0 release of VirtualiZarr.

1. [Appending new daily NOAA SST data to Icechunk](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V1/append/noaa-cdr-sst.ipynb)
2. [Parallel reference generation using Coiled Functions](https://github.com/zarr-developers/VirtualiZarr/blob/main/examples/V1/coiled/terraclimate.ipynb)
3. [Serverless parallel reference generation using Lithops](https://github.com/zarr-developers/VirtualiZarr/tree/main/examples/V1/virtualizarr-with-lithops)
4. [MUR SST Virtual and Zarr Icechunk Store Generation using Lithops](https://github.com/zarr-developers/VirtualiZarr/tree/main/examples/V1/mursst-icechunk-with-lithops)
