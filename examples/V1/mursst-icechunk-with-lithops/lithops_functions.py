"""
Lithops execution wrappers.

This module contains functions that wrap Lithops execution for various tasks.
"""

import logging
import subprocess

import lithops
from config import data_vars, lat_slice, lon_slice
from helpers import (
    find_label_for_range,
    get_mean,
    interval_df,
    open_and_read_data,
    xarray_open_icechunk,
)
from icechunk.distributed import merge_sessions
from models import Task
from repo import open_or_create_repo
from url_utils import list_mur_sst_files
from virtual_datasets import (
    concat_and_write_virtual_datasets,
    map_open_virtual_dataset,
)
from zarr_operations import (
    configure_zarr,
    handle_time_dimension,
    map_open_files,
    resize_data_array,
    write_data_to_zarr,
    xarray_open_mfdataset,
)


def get_fexec():
    """
    Get a Lithops executor.

    Returns:
        A Lithops executor
    """
    return lithops.FunctionExecutor(config_file="lithops.yaml", log_level=logging.INFO)


def lithops_write_virtual_references(
    start_date: str, end_date: str, append_dim: str = None
):
    """
    Write virtual references to Icechunk using Lithops.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        append_dim: The dimension to append to (optional)

    Returns:
        The result of the operation
    """
    uris = list_mur_sst_files(start_date, end_date)
    fexec = get_fexec()
    fexec.map_reduce(
        map_function=map_open_virtual_dataset,
        map_iterdata=uris,
        extra_args=(dict(filetype="dmrpp"),),
        reduce_function=concat_and_write_virtual_datasets,
        extra_args_reduce=(start_date, end_date, append_dim),
        spawn_reducer=100,
    )
    return fexec.get_result()


def lithops_write_zarr(start_date: str, end_date: str):
    """
    Write data to Zarr using Lithops.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format

    Returns:
        A message indicating the result of the operation
    """
    configure_zarr()
    files = list_mur_sst_files(start_date, end_date, dmrpp=False)

    # Open multi-file dataset with xarray
    fexec = get_fexec()
    fexec.map_reduce(
        map_function=map_open_files,
        map_iterdata=files,
        reduce_function=xarray_open_mfdataset,
        spawn_reducer=100,
    )
    ds = fexec.get_result()

    repo = open_or_create_repo()
    resize_session = repo.writable_session("main")
    resize_sessions = []
    with resize_session.allow_pickling():
        fexec.call_async(
            handle_time_dimension,
            data=dict(
                session=resize_session,
                start_date=start_date + " 09:00",
                end_date=end_date + " 09:00",
            ),
        )
        resize_time_session, dt_index_pairs = fexec.get_result()
        resize_sessions.append(resize_time_session)
        n_timesteps = len(dt_index_pairs)

        # Then resize data arrays
        fexec.map(
            map_function=resize_data_array,
            map_iterdata=data_vars,
            extra_args=(resize_session, n_timesteps),
        )
        resize_arrays_sessions = (
            fexec.get_result()
        )  # Wait for resize operations to complete
        resize_sessions.extend(resize_arrays_sessions)
    merge_sessions(resize_session, *resize_sessions)
    resize_commit_id = resize_session.commit(
        f"Resized time and data arrays for {start_date} to {end_date}"
    )
    print(f"Resize commit id: {resize_commit_id}")

    write_session = repo.writable_session("main")
    with write_session.allow_pickling():
        # Create tasks using datetime-index pairs
        tasks = []
        for var in data_vars:
            for dt, time_idx in dt_index_pairs:
                tasks.append(Task(var=var, dt=dt, time_idx=time_idx))

        fexec.map(
            map_function=write_data_to_zarr,
            map_iterdata=tasks,
            extra_args=(write_session, ds),
        )
        write_sessions = fexec.get_result()

    merge_sessions(write_session, *write_sessions)
    snapshot_id = write_session.commit(f"Wrote data {start_date} to {end_date}")
    return f"Wrote data to resized arrays, snapshot {snapshot_id}"


def write_to_icechunk(start_date: str, end_date: str, append_dim: str = None):
    """
    Write data to Icechunk.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        append_dim: The dimension to append to (optional)

    Returns:
        A message indicating the result of the operation
    """
    # Find which process is associated with start and end date
    process = find_label_for_range(start_date, end_date, interval_df)
    if process == "virtual_dataset":
        result = lithops_write_virtual_references(
            start_date=start_date, end_date=end_date, append_dim=append_dim
        )
        print(result)
    elif process == "zarr":
        result = lithops_write_zarr(start_date=start_date, end_date=end_date)
        print(result)
    return "Done"


def check_data_store_access(open_or_create_repo_func: callable = open_or_create_repo):
    """
    Check access to the data store.

    Args:
        open_or_create_repo_func: Function to open or create a repository

    Returns:
        The last time value in the dataset
    """
    xds = xarray_open_icechunk(open_or_create_repo_func)
    return xds["time"][-1]


def lithops_check_data_store_access():
    """
    Check access to the data store.

    Returns:
        A message indicating the result of the operation
    """
    fexec = get_fexec()
    fexec.call_async(
        func=check_data_store_access,
        data=dict(open_or_create_repo_func=open_or_create_repo),
    )
    print(fexec.get_result())


def lithops_calc_original_files_mean(
    start_date: str,
    end_date: str,
    lat_slice_arg: slice = lat_slice,
    lon_slice_arg: slice = lon_slice,
):
    """
    Calculate the mean of the original files.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        lat_slice_arg: The latitude slice
        lon_slice_arg: The longitude slice

    Returns:
        The mean value
    """
    from url_utils import list_mur_sst_files

    # map open and read data from selected space and time
    files = list_mur_sst_files(start_date, end_date, dmrpp=False)
    fexec = get_fexec()
    fexec.map_reduce(
        map_function=open_and_read_data,  # return array
        map_iterdata=files,
        extra_args=(lat_slice_arg, lon_slice_arg),
        reduce_function=get_mean,
    )
    print(fexec.get_result())


def calc_icechunk_store_mean(
    open_or_create_repo_func: callable = open_or_create_repo,
    start_date: str = None,
    end_date: str = None,
    lat_slice_arg: slice = lat_slice,
    lon_slice_arg: slice = lon_slice,
):
    """
    Calculate the mean of the Icechunk store.

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


def lithops_calc_icechunk_store_mean(start_date: str, end_date: str):
    """
    Calculate the mean of the Icechunk store.

    Args:
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format

    Returns:
        A message indicating the result of the operation
    """
    fexec = get_fexec()
    fexec.call_async(
        func=calc_icechunk_store_mean,
        data=dict(
            open_or_create_repo_func=open_or_create_repo,
            start_date=start_date,
            end_date=end_date,
            lat_slice_arg=lat_slice,
            lon_slice_arg=lon_slice,
        ),
    )
    print(fexec.get_result())


def list_installed_packages():
    """
    List installed packages using Lithops.

    Returns:
        A string containing the list of installed packages
    """
    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    return result.stdout


def lithops_list_installed_packages():
    """
    List installed packages.

    Returns:
        A message indicating the result of the operation
    """
    fexec = get_fexec()
    fexec.call_async(
        list_installed_packages,
        data={},
    )
    print(fexec.get_result())
