# Generate a virtual zarr dataset using lithops
## Note: This example uses a pre-2.0 release of VirtualiZarr

This example walks through how to create a virtual dataset from a collection of
netCDF files on s3 using lithops to open each file in parallel then concatenate
them into a single virtual dataset.

## Credits
Inspired by Pythia's cookbook: https://projectpythia.org/kerchunk-cookbook
by norlandrhagen.

Please, contribute improvements.



1. Set up a Python environment
```bash
conda create --name virtualizarr-lithops -y python=3.11
conda activate virtualizarr-lithops
pip install -r requirements.txt
```

2. Configure compute and storage backends for [lithops](https://lithops-cloud.github.io/docs/source/configuration.html).
The configuration in `lithops.yaml` uses AWS Lambda for [compute](https://lithops-cloud.github.io/docs/source/compute_config/aws_lambda.html) and AWS S3 for [storage](https://lithops-cloud.github.io/docs/source/storage_config/aws_s3.html).
To use those backends, simply edit `lithops.yaml` with your `bucket` and `execution_role`.

1. Build a runtime image for Cubed
```bash
export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
lithops runtime build -b aws_lambda -f Dockerfile_virtualizarr virtualizarr-runtime
```

1. Run the script
```bash
python virtualizarr-with-lithops.py
```

## Cleaning up
To rebuild the Lithops image, delete the existing one by running
```bash
lithops runtime delete -b aws_lambda -d virtualizarr-runtime
```
