"""
Data processing and validation.

This module contains functions for data processing, validation, and comparison.
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from config import lat_slice, lon_slice
from repo import open_or_create_repo


def xarray_open_icechunk(open_or_create_repo_func=open_or_create_repo):
    """
    Open an IceChunk repository as an xarray Dataset.

    Args:
        open_or_create_repo_func: Function to open or create a repository

    Returns:
        An xarray Dataset
    """
    # Configure Zarr for optimal performance
    zarr.config.set(
        {
            "async": {"concurrency": 100, "timeout": None},
            "threading": {"max_workers": None},
        }
    )
    repo = open_or_create_repo_func()
    session = repo.readonly_session("main")
    return xr.open_dataset(
        session.store, consolidated=False, zarr_format=3, engine="zarr"
    )


def lithops_check_data_store_access(open_or_create_repo_func=open_or_create_repo):
    """
    Check access to the data store.

    Args:
        open_or_create_repo_func: Function to open or create a repository

    Returns:
        The last time value in the dataset
    """
    xds = xarray_open_icechunk(open_or_create_repo_func)
    return xds["time"][-1]


def lithops_calc_icechunk_store_mean(
    open_or_create_repo_func=open_or_create_repo,
    start_date=None,
    end_date=None,
    lat_slice_arg=lat_slice,
    lon_slice_arg=lon_slice,
):
    """
    Calculate the mean of the IceChunk store.

    Args:
        open_or_create_repo_func: Function to open or create a repository
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        lat_slice_arg: The latitude slice
        lon_slice_arg: The longitude slice

    Returns:
        The mean value
    """
    xds = xarray_open_icechunk(open_or_create_repo_func)
    return (
        xds["analysed_sst"]
        .sel(time=slice(start_date, end_date), lat=lat_slice_arg, lon=lon_slice_arg)
        .mean()
        .values
    )


def open_and_read_data(file, lat_slice_arg=lat_slice, lon_slice_arg=lon_slice):
    """
    Open and read data from a file.

    Args:
        file: The file to open
        lat_slice_arg: The latitude slice
        lon_slice_arg: The longitude slice

    Returns:
        The data values
    """
    from config import fs_read

    ds = xr.open_dataset(fs_read.open(file), chunks={})
    return ds.analysed_sst.sel(lat=lat_slice_arg, lon=lon_slice_arg).values


def get_mean(values: np.ndarray):
    """
    Calculate the mean of an array.

    Args:
        values: The array to calculate the mean of

    Returns:
        The mean value
    """
    return np.nanmean(values)


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

# Convert dictionary to a Pandas DataFrame with IntervalIndex
interval_df = pd.DataFrame(
    [
        {
            "interval": pd.Interval(
                pd.Timestamp(start), pd.Timestamp(end), closed="both"
            ),
            "label": label,
        }
        for (start, end), label in date_process_dict.items()
    ]
)


def find_label_for_range(date_str1, date_str2, df=interval_df):
    """
    Find the corresponding label for two dates.

    Args:
        date_str1: The first date in YYYY-MM-DD format
        date_str2: The second date in YYYY-MM-DD format
        df: The DataFrame with intervals and labels

    Returns:
        The label for the date range
    """
    date1, date2 = pd.Timestamp(date_str1), pd.Timestamp(date_str2)

    # Find intervals where both dates are contained
    match = df[
        df["interval"].apply(lambda interval: date1 in interval and date2 in interval)
    ]
    if match.empty:
        raise ValueError(
            f"No matching interval found for dates {date_str1} and {date_str2}"
        )

    return match["label"].iloc[0] if not match.empty else None
