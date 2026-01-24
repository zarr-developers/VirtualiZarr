#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "virtualizarr",
#     "obstore",
#     "xarray",
#     "numpy",
#     "h5py",
# ]
# ///
"""
Virtualizing GOES-16 satellite data with a basic object store.

This example demonstrates virtualizing a GOES-16 file using a standard obstore
without any caching or parallel fetch optimizations. Compare this with
`goes_with_caching_stores.py` to see the performance difference.

## Usage

Run this example with uv:

    uv run --script examples/V2/goes_basic.py

To compare performance with the optimized version, run both scripts and compare
the "Virtualization time" printed at the end of each.

"""

import time

from obstore.store import from_url

import virtualizarr as vz
from virtualizarr.registry import ObjectStoreRegistry


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

    # --- Create the base object store (no caching or splitting) ---
    # This is a public S3 bucket, so we use skip_signature=True
    start_time = time.perf_counter()
    store = from_url(bucket, region="us-east-1", skip_signature=True)

    # --- Create the registry with the basic store ---
    registry = ObjectStoreRegistry({bucket: store})

    # --- Configure the HDF5 parser ---
    parser = vz.parsers.HDFParser(drop_variables=drop_variables)

    # --- Open the virtual dataset (timed) ---
    print(f"Opening virtual dataset from: {url}")

    vds = vz.open_virtual_dataset(
        url,
        registry=registry,
        parser=parser,
        loadable_variables=loadable_vars,
    )
    elapsed = time.perf_counter() - start_time

    # Show the size of virtual references vs actual data
    print(f"Size of virtual references: {vds.vz.nbytes:,} bytes")
    print(f"Size of actual data (if loaded): {vds.nbytes:,} bytes")
    print("-" * 10)

    print("-" * 10)
    print(f"Virtualization time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
