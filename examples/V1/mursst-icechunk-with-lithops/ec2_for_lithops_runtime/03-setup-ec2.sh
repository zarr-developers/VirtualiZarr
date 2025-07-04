#!/bin/bash

# Exit on error
set -e

echo "Updating system packages..."
sudo yum update -y

echo "Installing Python 3 and pip..."
sudo yum install -y python3 python3-pip

echo "Installing Docker..."
sudo yum install -y docker git

echo "Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "Adding current user to Docker group..."
sudo usermod -aG docker $USER

echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Verifying installations..."
python3 --version
pip3 --version
docker --version
uv --version

echo "Setup complete! Please log out and log back in to apply Docker group changes."

# lithops environment setup
git clone https://github.com/zarr-developers/Virtualizarr
cd Virtualizarr/
cd examples/mursst-icechunk-with-lithops/
uv venv virtualizarr-lithops
source virtualizarr-lithops/bin/activate
uv pip install -r requirements.txt
export LITHOPS_CONFIG_FILE=$(pwd)/lithops.yaml
lithops runtime build -b aws_lambda -f Dockerfile virtualizarr-runtime
