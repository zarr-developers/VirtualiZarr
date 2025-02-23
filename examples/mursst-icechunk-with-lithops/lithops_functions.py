import datetime
import logging
import sys
from typing import List, cast

import boto3
import fsspec
import icechunk
import lithops
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from virtualizarr import open_virtual_dataset

# Settings
fs_read = fsspec.filesystem("s3", anon=False, skip_instance_cache=True)
base_url = "s3://podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1/"
data_vars = ["analysed_sst", "analysis_error", "mask", "sea_ice_fraction"]
drop_vars = ["dt_1km_data", "sst_anomaly"]
bucket = "nasa-veda-scratch"
store_name = "MUR-JPL-L4-GLOB-v4.1-virtual-v5"

# TODO:
# - [ ] test small number of files with virtual refs
# - [ ] test small number of files with zarr
# - [ ] estimate memory runtime requirements
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
def map_open_virtual_dataset(fil):
    """Map function to open virtual datasets."""
    vds = open_virtual_dataset(
        fil,
        indexes={},
        filetype=["dmrpp"],
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


def write_virtual_results_to_icechunk(virtual_ds):
    repo = open_or_create_repo()
    session = repo.writable_session("main")
    store = session.store
    return virtual_ds.virtualize.to_icechunk(store, append_dim="time")


def concat_and_write_virtual_datasets(results):
    combined_vds = concat_virtual_datasets(results)
    return write_virtual_results_to_icechunk(combined_vds)


# Lithops wrapper funtion for virtual datasets
def lithops_write_virtual_references(start_date: datetime, end_date: datetime):
    files = list_mur_sst_files(start_date, end_date)
    futures = fexec.map_reduce(
        map_function=map_open_virtual_dataset,
        map_iterdata=files,
        reduce_function=concat_and_write_virtual_datasets,
        # map_runtime_memory=
        # reduce_runtime_memory=
        spawn_reducer=100,
    )
    futures.get_result()


# Zarr functions
def resize_array(var_name: str, group: zarr.Group, start_date: str, end_date: str):
    dt_index = pd.date_range(start=start_date, end=end_date, freq="1D")
    n_timesteps = len(dt_index)
    # current shape
    current_shape = group[var_name].shape
    group[var_name].resize((current_shape[0] + n_timesteps,) + current_shape[1:])
    if var_name == "time":
        reference_date = pd.Timestamp("1981-01-01 00:00:00")
        dt_index_seconds_since_1981 = (dt_index - reference_date).total_seconds()
        group[var_name][-n_timesteps:] = np.int32(dt_index_seconds_since_1981)
    return dt_index


def map_open_files(files):
    return [fs_read.open(f) for f in files]


def xarray_open_mfdataset(files):
    ds = xr.open_mfdataset(
        files, parallel=True, mask_and_scale=False, drop_variables=drop_vars
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


def write_data_to_zarr(
    task: dict, group: zarr.Group, ds: xr.Dataset, session: icechunk.Session
):
    var, dt, time_idx = task["var"], task["datetime"], task["time_idx"]
    data_array = ds[var].sel(time=dt)
    group = zarr.group(store=session.store, overwrite=False)
    current_array = cast(zarr.Array, group[var])
    current_array[time_idx, :, :] = data_array.values
    return None


def commit_to_icechunk(_, session: icechunk.Session, start_date: str, end_date: str):
    commit_response = session.commit(
        f"Distributed commit for {start_date} to {end_date}"
    )
    return commit_response


def lithops_write_zarr(start_date: datetime, end_date: datetime):
    files = list_mur_sst_files(start_date, end_date, dmrpp=False)
    repo = open_or_create_repo()
    session = repo.writable_session("main")
    store = session.store
    group = zarr.group(store=store, overwrite=False)

    # Resize arrays
    dt_index = fexec.map(
        map_function=resize_array,
        map_iterdata=data_vars + ["time"],
        data=dict(group=group, start_date=start_date, end_date=end_date),
    ).get_result()

    # Open multi-file dataset with xarray
    ds = fexec.map_reduce(
        map_function=map_open_files,
        map_iterdata=files,
        reduce_function=xarray_open_mfdataset,
        spawn_reducer=100,
    ).get_result()

    # Create a list of tasks to write the data to zarr
    # Each task has a variable, a datetime, and a time index
    tasks = []
    for var in data_vars:
        for time_idx, dt in enumerate(dt_index):
            tasks.append(
                {
                    "var": var,
                    "datetime": dt,
                    "time_idx": time_idx,
                }
            )

    futures = fexec.map_reduce(
        map_function=write_data_to_zarr,
        map_iterdata=tasks,
        extra_args=dict(group=group, ds=ds, session=session),
        reduce_function=commit_to_icechunk,
        extra_args_reduce=dict(
            session=session, start_date=start_date, end_date=end_date
        ),
        spawn_reducer=100,
    )
    futures.get_result()


def test_access(open_or_create_repo: callable):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    repo = open_or_create_repo()
    for snapshot in repo.ancestry(branch="main"):
        # this isn't working to write to STDOUT or the log, not sure why
        logging.info(f"{snapshot.message}, snapshot_id: {snapshot.id}")
    return None


fexec = lithops.FunctionExecutor(config_file="lithops.yaml", log_level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# Tests access
result = fexec.call_async(
    func=test_access, data={"open_or_create_repo": open_or_create_repo}
).result()
print(result)

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

for k, v in date_process_dict.items():
    if v == "virtual_dataset":
        lithops_write_virtual_references(start_date=k[0], end_date=k[1])
        print(result)
    elif v == "zarr":
        lithops_write_zarr(start_date=k[0], end_date=k[1])
        print(result)
