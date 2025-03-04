"""
URL utilities for MUR SST data processing.

This module contains functions for generating URLs and listing files.
"""

import datetime
from typing import List

import pandas as pd
from config import base_url


def make_url(date: datetime) -> str:
    """
    Create an S3 URL for a specific datetime.

    Args:
        date: The datetime to create a URL for

    Returns:
        The S3 URL for the specified datetime
    """
    date_string = date.strftime("%Y%m%d") + "090000"
    components = [
        base_url,
        f"{date_string}-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
    ]
    return "/".join(components)


def list_mur_sst_files(start_date: str, end_date: str, dmrpp: bool = True) -> List[str]:
    """
    List all files in S3 with a certain date prefix.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        dmrpp: Whether to return DMR++ URLs (default: True)

    Returns:
        A list of S3 URLs for the specified date range
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="1D")
    netcdf_urls = [make_url(date) for date in dates]
    if not dmrpp:
        return netcdf_urls
    return [f + ".dmrpp" for f in netcdf_urls]
