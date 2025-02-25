# Initialize dask config before importing
import argparse
import datetime
import logging
from dataclasses import dataclass
from typing import List, cast

import boto3
import fsspec
import icechunk
import lithops
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.distributed import merge_sessions

from virtualizarr import open_virtual_dataset

# Settings
fs_read = fsspec.filesystem("s3", anon=False, skip_instance_cache=True)
base_url = "s3://podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1"
data_vars = ["analysed_sst", "analysis_error", "mask", "sea_ice_fraction"]
drop_vars = ["dt_1km_data", "sst_anomaly"]
bucket = "nasa-veda-scratch"
store_name = "MUR-JPL-L4-GLOB-v4.1-virtual-v5"
lat_slice = slice(48.5, 48.7)
lon_slice = slice(-124.7, -124.5)

# Reset to a snapshot
# [print(f"{snapshot.message}, snapshot_id: {snapshot.id}") for snapshot in repo.ancestry(branch="main")]
# repo.reset_branch("main", "BLAH")

# TODO:
# - [x] test small number of files with virtual refs
# - [x] test small number of files with zarr
# - [ ] Write and test with validation function
# - [ ] estimate time to run + memory requirements for larger batches
# - [ ] run on larger batches of files


# NetCDF URL functions
def make_url(date: datetime) -> str:
    """Create an S3 URL for a specific dateime"""
    date_string = date.strftime("%Y%m%d") + "090000"
    components = [
        base_url,
        f"{date_string}-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
    ]
    return "/".join(components)


def list_mur_sst_files(start_date: str, end_date: str, dmrpp: bool = True) -> List[str]:
    """
    List all files in S3 with a certain date prefix
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="1D")
    netcdf_urls = [make_url(date) for date in dates]
    if not dmrpp:
        return netcdf_urls
    return [f + ".dmrpp" for f in netcdf_urls]


# Icechunk functions
def open_or_create_repo():
    # Config for repo storage
    session = boto3.Session()

    # Get the credentials from the session
    credentials = session.get_credentials()

    # Extract the actual key, secret, and token
    creds = credentials.get_frozen_credentials()
    storage_config = icechunk.s3_storage(
        bucket=bucket,
        prefix=f"icechunk/{store_name}",
        region="us-west-2",
        access_key_id=creds.access_key,
        secret_access_key=creds.secret_key,
        session_token=creds.token,
    )

    # Config for repo
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3", "s3://", icechunk.s3_store(region="us-west-2")
        )
    )

    # Config for repo virtual chunk credentials
    virtual_chunk_creds = icechunk.containers_credentials(
        s3=icechunk.s3_credentials(anonymous=False)
    )

    repo = icechunk.Repository.open_or_create(
        storage=storage_config,
        config=repo_config,
        virtual_chunk_credentials=virtual_chunk_creds,
    )
    return repo


# Virtual dataset functions
def map_open_virtual_dataset(uri):
    """Map function to open virtual datasets."""
    vds = open_virtual_dataset(
        uri,
        indexes={},
        filetype="dmrpp",
    )
    return vds.drop_vars(drop_vars, errors="ignore")


def concat_virtual_datasets(results):
    """Reduce to concat virtual datasets."""
    combined_vds = xr.concat(
        results,
        dim="time",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )
    return combined_vds


def write_virtual_results_to_icechunk(virtual_ds, start_date: str, end_date: str):
    repo = open_or_create_repo()
    session = repo.writable_session("main")

    # Check if store is already populated
    with session.allow_pickling():
        try:
            # Try to access any of the expected data variables to check if store is populated
            store_has_data = any(var in session.store for var in data_vars)
            # Only use append_dim if store already has data
            virtual_ds.virtualize.to_icechunk(
                session.store, append_dim="time" if store_has_data else None
            )
        except Exception:
            # If we can't check or there's an error, assume store is empty
            virtual_ds.virtualize.to_icechunk(session.store)

        return session.commit(f"Commit data {start_date} to {end_date}")


def concat_and_write_virtual_datasets(results, start_date: str, end_date: str):
    """Reduce to concat virtual datasets and write to icechunk."""
    combined_vds = concat_virtual_datasets(results)
    return write_virtual_results_to_icechunk(combined_vds, start_date, end_date)


# Lithops wrapper funtion for virtual datasets
def lithops_write_virtual_references(start_date: datetime, end_date: datetime):
    uris = list_mur_sst_files(start_date, end_date)
    fexec.map_reduce(
        map_function=map_open_virtual_dataset,
        map_iterdata=uris,
        reduce_function=concat_and_write_virtual_datasets,
        extra_args_reduce=(start_date, end_date),
        # map_runtime_memory=
        # reduce_runtime_memory=
        spawn_reducer=100,
    )
    return fexec.get_result()


# Zarr functions
def resize_data_array(var_name: str, session: icechunk.Session, n_timesteps: int):
    """Resize a data variable array."""
    group = zarr.group(store=session.store, overwrite=False)
    current_shape = group[var_name].shape
    group[var_name].resize((current_shape[0] + n_timesteps,) + current_shape[1:])
    return session


def handle_time_dimension(session: icechunk.Session, start_date: str, end_date: str):
    """Handle time dimension and return datetime-index pairs."""
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


def map_open_files(file):
    return fs_read.open(file)


def xarray_open_mfdataset(files):
    ds = xr.open_mfdataset(
        files, mask_and_scale=False, drop_variables=drop_vars, chunks={}
    )
    ds["analysed_sst"] = ds["analysed_sst"].chunk({"time": 1, "lat": 1023, "lon": 2047})
    ds["analysis_error"] = ds["analysis_error"].chunk(
        {"time": 1, "lat": 1023, "lon": 2047}
    )
    ds["mask"] = ds["mask"].chunk({"time": 1, "lat": 1447, "lon": 2895})
    ds["sea_ice_fraction"] = ds["sea_ice_fraction"].chunk(
        {"time": 1, "lat": 1447, "lon": 2895}
    )
    return ds


@dataclass
class Task:
    var: str
    dt: str
    time_idx: int


def write_data_to_zarr(task: Task, session: icechunk.Session, ds: xr.Dataset):
    group = zarr.group(store=session.store, overwrite=False)
    var, dt, time_idx = task.var, task.dt, task.time_idx
    data_array = ds[var].sel(time=dt)
    current_array = cast(zarr.Array, group[var])
    # where we actually write the data
    current_array[time_idx, :, :] = data_array.values
    return session


def lithops_write_zarr(start_date: str, end_date: str):
    zarr.config.set(
        {
            "async": {"concurrency": 100, "timeout": None},
            "threading": {"max_workers": None},
        }
    )
    files = list_mur_sst_files(start_date, end_date, dmrpp=False)
    # Open multi-file dataset with xarray
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


def lithops_check_data_store_access(open_or_create_repo: callable):
    repo = open_or_create_repo()
    session = repo.readonly_session("main")
    xds = xr.open_dataset(
        session.store, consolidated=False, zarr_format=3, engine="zarr"
    )
    return xds["analysed_sst"].sel(lat=lat_slice, lon=lon_slice).mean().values


def open_and_read_data(file, lat_slice, lon_slice):
    ds = xr.open_dataset(fs_read.open(file), chunks={})
    return ds.analysed_sst.sel(lat=lat_slice, lon=lon_slice).values


def get_mean(values: np.ndarray):
    return np.nanmean(values)


def lithops_check_original_files(
    start_date: str, end_date: str, lat_slice: tuple, lon_slice: tuple
):
    # map open and read data from selected space and time
    files = list_mur_sst_files(start_date, end_date, dmrpp=False)
    # fexec.map_reduce(
    #     map_function=open_and_read_data, # return array
    #     map_iterdata=files,
    #     extra_args=(lat_slice, lon_slice),
    #     reduce_function=get_mean,
    # )
    fexec.map(
        map_function=open_and_read_data,  # return array
        map_iterdata=files,
        extra_args=(lat_slice, lon_slice),
        # reduce_function=get_mean,
    )
    return fexec.get_result()


# Create a list of date ranges
date_process_dict = {
    ("2002-06-30", "2003-09-10"): "virtual_dataset",
    ("2003-09-11", "2003-09-11"): "zarr",
    ("2003-09-12", "2021-02-19"): "virtual_dataset",
    ("2021-02-20", "2021-02-21"): "zarr",
    ("2021-02-22", "2021-12-23"): "virtual_dataset",
    ("2021-12-24", "2022-01-26"): "zarr",
    ("2022-01-27", "2022-11-08"): "virtual_dataset",
    ("2022-11-09", "2022-11-09"): "zarr",
    ("2022-11-19", "2023-02-23"): "virtual_dataset",
    ("2023-02-24", "2023-02-28"): "zarr",
    ("2023-03-01", "2023-04-21"): "virtual_dataset",
    ("2023-04-22", "2023-04-22"): "zarr",
    ("2023-04-23", "2023-09-03"): "virtual_dataset",
}

test_date_process_dict = {
    ("2002-07-06", "2002-07-07"): "zarr",
}

fexec = lithops.FunctionExecutor(config_file="lithops.yaml", log_level=logging.INFO)


# Wrapper functions for calling lithops
## Main function - write to icechunk
def write_to_icechunk():
    for k, v in test_date_process_dict.items():
        if v == "virtual_dataset":
            result = lithops_write_virtual_references(start_date=k[0], end_date=k[1])
            print(result)
        elif v == "zarr":
            result = lithops_write_zarr(start_date=k[0], end_date=k[1])
            print(result)
    return "Done"


## Test data store access
def check_data_store_access():
    fexec.call_async(
        func=lithops_check_data_store_access,
        data={"open_or_create_repo": open_or_create_repo},
    )
    print(fexec.get_result())


## Test original files
def check_original_files():
    result = lithops_check_original_files(
        start_date="2002-06-30",
        end_date="2002-07-07",
        lat_slice=lat_slice,
        lon_slice=lon_slice,
    )
    print(result)


## For debugging the environment
def lithops_list_installed_packages():
    import subprocess

    result = subprocess.run(["pip", "list"], capture_output=True, text=True)
    return result.stdout


def list_installed_packages():
    fexec.call_async(
        lithops_list_installed_packages,
        data={},
    )
    print(fexec.get_result())


def parse_args():
    parser = argparse.ArgumentParser(description="Run lithops functions.")
    parser.add_argument(
        "function",
        choices=[
            "write_to_icechunk",
            "check_data_store_access",
            "check_original_files",
            "list_installed_packages",
        ],
        help="The function to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.function == "write_to_icechunk":
        write_to_icechunk()
    elif args.function == "check_data_store_access":
        check_data_store_access()
    elif args.function == "check_original_files":
        check_original_files()
    elif args.function == "list_installed_packages":
        list_installed_packages()
