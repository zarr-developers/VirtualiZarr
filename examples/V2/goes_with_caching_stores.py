#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "virtualizarr @ git+https://github.com/maxrjones/VirtualiZarr@obspec-utils-readablefile-protocol",
#     "obstore",
#     "obspec-utils",
#     "xarray",
#     "numpy",
#     "h5py",
# ]
# ///
"""
Virtualizing GOES-16 satellite data with optimized store wrappers.

This example demonstrates how to use CachingReadableStore and SplittingReadableStore
from obspec-utils to optimize file access when virtualizing remote data files.

## Store Wrappers

When virtualizing remote files, VirtualiZarr needs to read metadata from each file.
For files stored in cloud object storage, this can involve many small HTTP requests,
which are slow due to network latency.

Two store wrappers from obspec-utils help optimize this:

1. **SplittingReadableStore**: Accelerates downloading large files by splitting a
   single `get()` request into multiple parallel `get_ranges()` calls. This takes
   advantage of cloud storage's high per-request bandwidth.

2. **CachingReadableStore**: Caches entire files in memory after first access.
   Subsequent accesses (including range requests) are served from the cache.
   Uses LRU eviction when the cache exceeds its maximum size.

## Composition Pattern

These wrappers are designed to be composed together:

```
SplittingReadableStore  ->  CachingReadableStore  ->  VirtualiZarr
     (fast fetch)              (cache result)         (virtualize)
```

The SplittingReadableStore fetches the file quickly via parallel requests,
then CachingReadableStore stores the result for reuse during virtualization.

## Usage

Run this example with uv:

    uv run --script examples/V2/goes_with_caching_stores.py

"""

import time

from obspec_utils.cache import CachingReadableStore
from obspec_utils.obspec import BufferedStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from obspec_utils.splitting import SplittingReadableStore
from obstore.store import from_url

import virtualizarr as vz


def per_band_var_names(var_name: str) -> list[str]:
    """Generate variable names for all 16 GOES ABI bands."""
    return [f"{var_name}_C{i:02}" for i in range(1, 17)]


def main():
    # --- Configuration ---
    bucket = "s3://noaa-goes16"
    url = (
        "s3://noaa-goes16/ABI-L2-MCMIPF/2024/099/18/"
        "OR_ABI-L2-MCMIPF-M6_G16_s20240991800204_e20240991809524_c20240991810005.nc"
    )

    # Variables to drop (band statistics we don't need for this example)
    drop_variables = (
        per_band_var_names("band_id")
        + per_band_var_names("min_reflectance_factor")
        + per_band_var_names("max_reflectance_factor")
        + per_band_var_names("mean_reflectance_factor")
        + per_band_var_names("std_dev_reflectance_factor")
        + per_band_var_names("min_brightness_temperature")
        + per_band_var_names("max_brightness_temperature")
        + per_band_var_names("mean_brightness_temperature")
        + per_band_var_names("std_dev_brightness_temperature")
        + per_band_var_names("outlier_pixel_count")
    )

    # Variables to load into memory (low-dimensional coordinates and metadata)
    loadable_vars = [
        "y",
        "x",
        "t",
        "band",
        "x_image",
        "y_image",
        "x_image_bounds",
        "y_image_bounds",
        "time_bounds",
        "goes_imager_projection",
        "nominal_satellite_subpoint_lat",
        "nominal_satellite_subpoint_lon",
        "nominal_satellite_height",
        "geospatial_lat_lon_extent",
        "percent_uncorrectable_GRB_errors",
        "percent_uncorrectable_L0_errors",
        "dynamic_algorithm_input_data_container",
        "algorithm_product_version_container",
    ] + per_band_var_names("band_wavelength")

    # --- Create the base object store ---
    # This is a public S3 bucket, so we use skip_signature=True
    start_time = time.perf_counter()
    base_store = from_url(bucket, region="us-east-1", skip_signature=True)

    # --- Wrap with SplittingReadableStore for parallel fetching ---
    # This splits large file fetches into parallel range requests.
    # Default settings (12 MB chunks, 18 concurrent requests) work well for cloud storage.
    splitting_store = SplittingReadableStore(base_store)

    # --- Wrap with CachingReadableStore for in-memory caching ---
    # This caches entire files after first access. When VirtualiZarr reads
    # metadata from different parts of the file, all reads are served from cache.
    # Default max_size is 256 MB, which is sufficient for most single-file cases.
    caching_store = CachingReadableStore(splitting_store, max_size=512 * 1024 * 1024)

    # --- Create the registry with the wrapped store ---
    registry = ObjectStoreRegistry({bucket: caching_store})

    # --- Configure the HDF5 parser ---
    parser = vz.parsers.HDFParser(
        drop_variables=drop_variables, reader_factory=BufferedStoreReader
    )

    # --- Open the virtual dataset (timed) ---
    print(f"Opening virtual dataset from: {url}")
    print(f"Cache size before: {caching_store.cache_size} bytes")

    vds = vz.open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        loadable_variables=loadable_vars,
    )
    elapsed = time.perf_counter() - start_time

    print(f"Cache size after: {caching_store.cache_size:,} bytes")
    print(f"Cached paths: {caching_store.cached_paths}")
    print("-" * 10)

    # Show the size of virtual references vs actual data
    print(f"Size of virtual references: {vds.vz.nbytes:,} bytes")
    print(f"Size of actual data (if loaded): {vds.nbytes:,} bytes")
    print("-" * 10)

    # --- Cleanup ---
    # Clear the cache when done (optional - will be garbage collected anyway)
    caching_store.clear_cache()
    print("-" * 10)
    print(f"Virtualization time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
