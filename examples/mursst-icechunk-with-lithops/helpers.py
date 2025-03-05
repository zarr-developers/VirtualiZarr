"""
Helpers.
"""

import numpy as np
import pandas as pd
import xarray as xr
from config import date_process_dict, lat_slice, lon_slice
from repo import open_or_create_repo
from zarr_operations import configure_zarr


def xarray_open_icechunk(open_or_create_repo_func: callable = open_or_create_repo):
    """
    Open an Icechunk repository as an xarray Dataset.

    Args:
        open_or_create_repo_func: Function to open or create a repository

    Returns:
        An xarray Dataset
    """
    # Configure Zarr for optimal performance
    configure_zarr()
    repo = open_or_create_repo_func()
    session = repo.readonly_session("main")
    return xr.open_dataset(
        session.store, consolidated=False, zarr_format=3, engine="zarr"
    )


def open_and_read_data(
    file: str, lat_slice_arg: slice = lat_slice, lon_slice_arg: slice = lon_slice
):
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
