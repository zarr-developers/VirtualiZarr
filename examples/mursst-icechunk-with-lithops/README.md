# Generate a virtual zarr dataset using lithops

This example walks through how to create a virtual dataset from a collection of
netCDF files on s3 using lithops to open each file in parallel then concatenate
them into a single virtual dataset.

## Credits

Inspired by Pythia's cookbook: https://projectpythia.org/kerchunk-cookbook
by norlandrhagen.

Please, contribute improvements.

1. Set up a Python environment with uv:

```sh
uv venv virtualizarr-lithops
source virtualizarr-lithops/bin/activate
uv pip install -r requirements.txt
```

2. Configure compute and storage backends for [lithops](https://lithops-cloud.github.io/docs/source/configuration.html).
   The configuration in `lithops.yaml` supports AWS Lambda for compute and AWS S3 for storage.

   - Edit `lithops.yaml` and set `backend: aws_lambda`
   - Configure your `bucket` and `execution_role`
   - Build the runtime:
     ```bash
     export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
     lithops runtime build -b aws_lambda -f Dockerfile virtualizarr-runtime
     ```

3. Test it's working

```bash
python test_lithops.py
```

4. Run the script

```bash
python lithops_functions.py <function_name> --start_date <start_date> --end_date <end_date> --append_dim time
```

## Cleaning up

### Lambda Backend

To rebuild the Lithops Lambda runtime image, delete the existing one:

```bash
lithops runtime delete -b aws_lambda -d virtualizarr-runtime
```
