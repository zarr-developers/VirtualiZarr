# Generating a combined virtual + native Zarr store for MUR SST using icehunk and lithops

This example demonstrates how to create a virtual and native Zarr dataset from a collection of
netCDF files on s3. This example uses lithops for various map and map reduce operations involved in reading and writing the data as either virtual references or native Zarr. [Icechunk](https://icechunk.io) is used as the storage engine.

## Credits

Inspired by the [Generate a virtual zarr dataset using lithops](./virtualizarr-lithops/) example by @thodson-usgs.

## Setup

1. Set up a Python environment with uv:

```sh
uv venv virtualizarr-lithops --python 3.11
source virtualizarr-lithops/bin/activate
uv pip install -r requirements.txt
```

2. Configure compute and storage backends for [lithops](https://lithops-cloud.github.io/docs/source/configuration.html) in `lithops.yaml`. Note the `aws_lambda: runtime:` value should match the runtime name used in the `lithops runtime build` command below.

3. Build the lambda runtime:

```bash
export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
lithops runtime build -b aws_lambda -f Dockerfile vz-runtime
```

> [!IMPORTANT]  
> To rebuild the lithops lambda runtime image, you first need to delete the existing one:
>```bash
>lithops runtime delete -b aws_lambda -d vz-runtime
>```
>You can also update the lambda configuration to use to the `latest` tag in the image URI declaration, but you will need to `Deploy image` to the lambda whenever there are changes.

4. Test it's working

```bash
python test_lithops.py
```

## Reading, writing and validating the MUR SST icechunk store.

`lithops_functions.py` includes a number of functions for writing data and validating the store. There is customization as to which dates shoudl be written as native Zarr and which should be written as virtual data, which is more fully explained in this [MUR SST Icechunk Dataset Design Document](https://github.com/earth-mover/icechunk-nasa/blob/main/design-docs/mursst-virtual-icechunk-store.md).

These functions are intended to be used as follows:

1. (Optional) Check data store access:

```bash
python lithops_functions.py check_data_store_access
```

2. Write data for some datetime range, omitting `append_dim` if you are initiating the data store.

```bash
python lithops_functions.py write_to_icechunk --start_date <start_date> --end_date <end_date> --append_dim time
```

3. Validate the data written. Because it can take awhile to read from the original files, I usually do this for a small subset of the data, say 10 days.

The following script calculates the mean for a small spatial subset using the original files and `xr.open_mfdataset`.

```bash
python lithops_functions.py calc_original_files_mean --start_date 2021-01-01 --end_date 2021-01-11 
```

the output value should be compared with the output of the icechunk store:

```bash
python lithops_functions.py calc_icechunk_store_mean --start_date 2021-01-01 --end_date 2021-01-11 
```

# TODOs:

- [ ] Use a uv environment in the `Dockerfile` rather than installing packages globally
- [ ] Document how to build runtime using ec2
- [ ] Update documentation in earth-mover/icechunk-nasa repo
