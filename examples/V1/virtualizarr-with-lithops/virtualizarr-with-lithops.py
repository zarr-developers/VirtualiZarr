# Use lithops to create a virtual dataset from a collection of necdf files on s3.
# Note: This example uses a pre-2.0 release of VirtualiZarr

# Inspired by Pythia's cookbook: https://projectpythia.org/kerchunk-cookbook
# by norlandrhagen.
#
# Please, contribute improvements.

import fsspec
import lithops
import xarray as xr

from virtualizarr import open_virtual_dataset

# to demonstrate this workflow, we will use a collection of netcdf files from the WRF-SE-AK-AR5 project.
fs_read = fsspec.filesystem("s3", anon=True, skip_instance_cache=True)
files_paths = fs_read.glob("s3://wrf-se-ak-ar5/ccsm/rcp85/daily/2060/*")
file_pattern = sorted(["s3://" + f for f in files_paths])

# optionally, truncate file_pattern while debugging
# file_pattern = file_pattern[:4]

print(f"{len(file_pattern)} file paths were retrieved.")


def map_references(fil):
    """Map function to open virtual datasets."""
    vds = open_virtual_dataset(
        fil,
        indexes={},
        loadable_variables=["Time"],
        cftime_variables=["Time"],
    )
    return vds


def reduce_references(results):
    """Reduce to concat virtual datasets."""
    combined_vds = xr.combine_nested(
        results,
        concat_dim=["Time"],
        coords="minimal",
        compat="override",
    )
    return combined_vds


fexec = lithops.FunctionExecutor(config_file="lithops.yaml")

futures = fexec.map_reduce(
    map_references,
    file_pattern,
    reduce_references,
    spawn_reducer=100,
)

ds = futures.get_result()

# write out the virtual dataset to a kerchunk json
ds.virtualize.to_kerchunk("combined.json", format="json")
