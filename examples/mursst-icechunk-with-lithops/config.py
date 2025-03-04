"""
Configuration settings for MUR SST data processing.

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

# Spatial subset configuration
lat_slice = slice(48.5, 48.7)
lon_slice = slice(-124.7, -124.5)
