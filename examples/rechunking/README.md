# Example rechunking workflow

1. Set up a Python environment
```bash
conda create --name virtualizarr-rechunk -y python=3.11
conda activate virtualizarr-rechunk
pip install -r requirements.txt
```

1. Set up cubed executor by following https://github.com/cubed-dev/cubed/blob/main/examples/lithops/aws/README.md

1. Build a runtime image for Cubed
```bash
lithops runtime build -b aws_lambda -f Dockerfile_aws_lambda virtualizarr-runtime
```
