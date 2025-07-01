"""
Zarr operations.

This module contains functions for working with Zarr arrays.
"""

from typing import cast

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from config import mursst_var_chunks, zarr_concurrency
from models import Task


def resize_data_array(var_name: str, session: icechunk.Session, n_timesteps: int):
    """
    Resize a data variable array.

    Args:
        var_name: The name of the variable to resize
        session: The IceChunk session
        n_timesteps: The number of timesteps to add

    Returns:
        The updated session
    """
    group = zarr.group(store=session.store, overwrite=False)
    current_shape = group[var_name].shape
    group[var_name].resize((current_shape[0] + n_timesteps,) + current_shape[1:])
    return session


def handle_time_dimension(session: icechunk.Session, start_date: str, end_date: str):
    """
    Handle time dimension and return datetime-index pairs.

    Args:
        session: The Icechunk session
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format

    Returns:
        A tuple containing the updated session and a list of datetime-index pairs
    """
    group = zarr.group(store=session.store, overwrite=False)
    dt_index = pd.date_range(start=start_date, end=end_date, freq="1D")
    n_timesteps = len(dt_index)
    current_time_length = group["time"].shape[0]

    # Resize time array
    group["time"].resize((current_time_length + n_timesteps,))

    # Update time values
    reference_date = pd.Timestamp("1981-01-01 00:00:00")
    dt_index_seconds_since_1981 = (dt_index - reference_date).total_seconds()
    group["time"][-n_timesteps:] = np.int32(dt_index_seconds_since_1981)

    # Return list of (datetime, index) pairs
    return (
        session,
        [(dt, current_time_length + idx) for idx, dt in enumerate(dt_index)],
    )


def write_data_to_zarr(task: Task, session: icechunk.Session, ds: xr.Dataset):
    """
    Write data to Zarr array.

    Args:
        task: The task containing variable, datetime, and time index
        session: The Icechunk session
        ds: The xarray Dataset containing the data

    Returns:
        The updated session
    """
    group = zarr.group(store=session.store, overwrite=False)
    var, dt, time_idx = task.var, task.dt, task.time_idx
    data_array = ds[var].sel(time=dt)
    current_array = cast(zarr.Array, group[var])
    # where we actually write the data
    current_array[time_idx, :, :] = data_array.values
    return session


def configure_zarr():
    """
    Configure Zarr settings for optimal performance.
    """
    zarr.config.set(
        {
            "async": {"concurrency": zarr_concurrency, "timeout": None},
            "threading": {"max_workers": None},
        }
    )


def map_open_files(file: str):
    """
    Map function to open files.

    Args:
        file: The file to open

    Returns:
        An opened file object
    """
    from config import fs_read

    return fs_read.open(file)


def xarray_open_mfdataset(files: list[str]):
    """
    Open multiple files as an xarray Dataset.

    Args:
        files: A list of file objects

    Returns:
        An xarray Dataset
    """
    from config import drop_vars

    ds = xr.open_mfdataset(
        files, mask_and_scale=False, drop_variables=drop_vars, chunks={}
    )
    for var, chunks in mursst_var_chunks.items():
        ds[var] = ds[var].chunk(chunks)
    return ds
