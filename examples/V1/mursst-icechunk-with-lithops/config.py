"""
Configuration settings for MUR SST data processing.
Note: This example uses a pre-2.0 release of VirtualiZarr

This module contains all the configuration settings and constants used
throughout the package.
"""

import fsspec

# S3 filesystem for reading data
fs_read = fsspec.filesystem("s3", anon=False, skip_instance_cache=True)

# Data source configuration
base_url = "s3://podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1"
data_vars = ["analysed_sst", "analysis_error", "mask", "sea_ice_fraction"]
drop_vars = ["dt_1km_data", "sst_anomaly"]

# Storage configuration
bucket = "nasa-eodc-scratch"
store_name = "MUR-JPL-L4-GLOB-v4.1-virtual-v1"
directory = "test"

# Spatial subset configuration
lat_slice = slice(48.5, 48.7)
lon_slice = slice(-124.7, -124.5)

# Date range processing dictionary
date_process_dict = {
    ("2002-06-30", "2003-09-10"): "virtual_dataset",
    ("2003-09-11", "2003-09-11"): "zarr",
    ("2003-09-12", "2021-02-19"): "virtual_dataset",
    ("2021-02-20", "2021-02-21"): "zarr",
    ("2021-02-22", "2021-12-23"): "virtual_dataset",
    ("2021-12-24", "2022-01-26"): "zarr",
    ("2022-01-27", "2022-11-08"): "virtual_dataset",
    ("2022-11-09", "2022-11-09"): "zarr",
    ("2022-11-10", "2023-02-23"): "virtual_dataset",
    ("2023-02-24", "2023-02-28"): "zarr",
    ("2023-03-01", "2023-04-21"): "virtual_dataset",
    ("2023-04-22", "2023-04-22"): "zarr",
    ("2023-04-23", "2023-09-03"): "virtual_dataset",
}

zarr_concurrency = 4

mursst_var_chunks = {
    "analysed_sst": {"time": 1, "lat": 1023, "lon": 2047},
    "analysis_error": {"time": 1, "lat": 1023, "lon": 2047},
    "mask": {"time": 1, "lat": 1447, "lon": 2895},
    "sea_ice_fraction": {"time": 1, "lat": 1447, "lon": 2895},
}
