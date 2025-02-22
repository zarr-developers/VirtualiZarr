import datetime
from typing import List

import fsspec
import icechunk
import lithops
import pandas as pd
import xarray as xr

from virtualizarr import open_virtual_dataset

# to demonstrate this workflow, we will use a collection of netcdf files from the WRF-SE-AK-AR5 project.
fs_read = fsspec.filesystem("s3", anon=True, skip_instance_cache=True)
base_url = "s3://podaac-ops-cumulus-protected/MUR-JPL-L4-GLOB-v4.1/"
drop_vars = ["dt_1km_data", "sst_anomaly"]


def make_url(date: datetime) -> str:
    """Create an S3 URL for a specific dateime"""
    date_string = date.strftime("%Y%m%d") + "090000"
    components = [
        base_url,
        f"{date_string}-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc",
    ]
    return "/".join(components)


def list_mur_sst_files(start_date: str, end_date: str) -> List[str]:
    """
    List all files in S3 with a certain date prefix
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="1D")
    return [make_url(date) for date in dates]


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


fexec = lithops.FunctionExecutor(config_file="lithops.yaml")

storage = icechunk.local_filesystem_storage("./mursst-icechunk")

config = icechunk.RepositoryConfig.default()

config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer("s3", "s3://", icechunk.s3_store(region="us-west-2"))
)

credentials = icechunk.containers_credentials(
    s3=icechunk.s3_credentials(anonymous=False, region="us-west-2")
)

repo = icechunk.Repository.open_or_create(storage, config, credentials)


# A function for map reducing a list of files as virtual references and writing to icechunk
def write_virtual_references(start_date: datetime, end_date: datetime):
    files = list_mur_sst_files(start_date, end_date)
    futures = fexec.map_reduce(
        map_function=map_open_virtual_dataset,
        map_iterdata=files,
        reduce_function=concat_virtual_datasets,
        # map_runtime_memory=
        # reduce_runtime_memory=
        spawn_reducer=100,
    )
    virtual_ds = futures.get_result()
    session = repo.writable_session("main")
    store = session.store
    virtual_ds.virtualize.to_icechunk(store, append_dim="time")
