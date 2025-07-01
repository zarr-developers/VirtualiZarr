# Lithops Package for MUR SST Data Processing
## Note: This example uses a pre-2.0 release of VirtualiZarr

This package provides functionality for processing MUR SST (Multi-scale Ultra-high Resolution Sea Surface Temperature) data using [Lithops](https://lithops-cloud.github.io/), a framework for serverless computing.

## Environment + Lithops Setup

1. Set up a Python environment. The below example uses [`uv`](https://docs.astral.sh/uv/), but other environment managers should work as well:

```sh
uv venv virtualizarr-lithops --python 3.11
source virtualizarr-lithops/bin/activate
uv pip install -r requirements.txt
```

2. Follow the [AWS Lambda Configuration](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html#configuration) instructions, unless you already have an appropriate AWS IAM role to use.

3. Follow the [AWS Credential setup](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html#aws-credential-setup) instructions.

4. Check and modify as necessary compute and storage backends for [lithops](https://lithops-cloud.github.io/docs/source/configuration.html) in `lithops.yaml`.


5. Build the lithops lambda runtime if it does not exist in your target AWS environment.
```bash
export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
lithops runtime build -b aws_lambda -f Dockerfile vz-runtime
```

For various reasons, you may want to build the lambda runtime on EC2 (docker can be a resource hog and pushing to ECR is faster, for example). If you wish to use EC2, please see the scripts in `ec2_for_lithops_runtime/` in this directory.

> [!IMPORTANT]
> If the runtime was created with a different IAM identity, an appropriate `user_id` will need to be included in the lithops configuration under `aws_lamda`.

> [!TIP]
> You can configure the AWS Lambda architecture via the `architecture` key under `aws_lambda` in the lithops configuration file.


6. (Optional) To rebuild the Lithops Lambda runtime image, delete the existing one:

```bash
lithops runtime delete -b aws_lambda -d virtualizarr-runtime
```

## Package Structure

The package is organized into the following modules:

- `__init__.py`: Package initialization and exports
- `config.py`: Configuration settings and constants
- `models.py`: Data models and structures
- `url_utils.py`: URL generation and file listing
- `repo.py`: Icechunk repository management
- `virtual_datasets.py`: Virtual dataset operations
- `zarr_operations.py`: Zarr array operations
- `helpers.py`: Data helpers
- `lithops_functions.py`: Lithops execution wrappers
- `cli.py`: Command-line interface

## Usage

### Command-line Interface

The package provides a command-line interface for running various functions:

```bash
python main.py <function> [options]
```

Available functions:

- `write_to_icechunk`: Write data to Icechunk
- `check_data_store_access`: Check access to the data store
- `calc_icechunk_store_mean`: Calculate the mean of the Icechunk store
- `calc_original_files_mean`: Calculate the mean of the original files
- `list_installed_packages`: List installed packages

Options:

- `--start_date`: Start date for data processing (YYYY-MM-DD)
- `--end_date`: End date for data processing (YYYY-MM-DD)
- `--append_dim`: Append dimension for writing to Icechunk

### Examples

#### Writing Data to Icechunk

```bash
python main.py write_to_icechunk --start_date 2022-01-01 --end_date 2022-01-02
```

#### Calculating the Mean of the Icechunk Store

```bash
python main.py calc_icechunk_store_mean --start_date 2022-01-01 --end_date 2022-01-31
```

#### Checking Data Store Access

```bash
python main.py check_data_store_access
```

## Programmatic Usage

You can also use the package programmatically:

```python
from lithops_functions import write_to_icechunk

# Write data to Icechunk
write_to_icechunk(start_date="2022-01-01", end_date="2022-01-31")
```

## Testing

To test the package, you can use the provided test functions:

```bash
python main.py check_data_store_access
```

This will verify that the package can access the data store.
