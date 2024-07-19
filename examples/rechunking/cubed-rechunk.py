# Example rechunking flow using VirtualiZarr and cubed.

import fsspec
import lithops
import xarray as xr

from virtualizarr import open_virtual_dataset
from cubed.primitive.rechunk import rechunk

# example workflow based on Pythia's kerchunk cookbook:
# https://projectpythia.org/kerchunk-cookbook/notebooks/foundations/02_kerchunk_multi_file.html
fs_read = fsspec.filesystem("s3", anon=True, skip_instance_cache=True)
files_paths = fs_read.glob("s3://wrf-se-ak-ar5/ccsm/rcp85/daily/2060/*")
file_pattern = sorted(["s3://" + f for f in files_paths])

# truncate file_pattern while debugging
file_pattern = file_pattern[:4]

print(f"{len(file_pattern)} file paths were retrieved.")


def map_reference(fil):
    """ Map 
    """
    vds = open_virtual_dataset(fil,
                               indexes={},
                               loadable_variables=['Time'],
                               cftime_variables=['Time'],
                               )
    return vds


def reduce_reference(results):
    """
    """
    combined_vds = xr.combine_nested(
        results,
        concat_dim=['Time'],
        coords='minimal',
        compat='override',
    )
    # possibly write parquet to s3 here
    return combined_vds


fexec = lithops.FunctionExecutor()  # config=lambda_config

futures = fexec.map_reduce(
    map_reference,
    file_pattern,
    reduce_reference,
    spawn_reducer=100
    )

ds = futures.get_result()

ds.virtualize.to_kerchunk('combined.json', format='json')

# in notebooks, open_dataset must be caching the json, such that changes 
# to the json are not propogated until the kernel is restarted
combined_ds = xr.open_dataset('combined.json', engine="kerchunk")
combined_ds['Time'].attrs = {}  # to_zarr complains about attrs


source_chunks = {'Time': 1, 'south_north': 250, 'west_east': 320}
target_chunks = {'Time': 5, 'south_north': 25, 'west_east': 32}

combined_chunked = combined_ds.chunk(
    chunks=source_chunks,
)

# rechunk requires shape attr, so can't pass full Dataset
rechunk(
    combined_chunked['TMAX'],
    target_chunks=target_chunks,
    source_array_name='virtual',
    int_array_name='temp',
    allowed_mem=2000,
    reserved_mem=1000,
    target_store="test.zarr",
)
