# CCMP Recipes: VirtualiZarr + Icechunk with NASA Earthdata

Notebook demonstrating how to virtualize the ~750 GB CCMP wind dataset from PO.DAAC, write to Icechunk, parallelize with Dask, and append new data. Based on the Po.DAAC tutorial by Dean Henze at https://podaac.github.io/tutorials/notebooks/Advanced_cloud/virtualizarr_recipes.html.

## Prerequisites

- NASA Earthdata account: https://urs.earthdata.nasa.gov/
- Credentials via `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` env vars or `~/.netrc`
- Must run from **AWS us-west-2** for direct S3 access
- Recommended: `m6i.4xlarge` EC2 instance (16 CPUs, 64 GiB) for parallel sections

## Setup

Install [uv](https://docs.astral.sh/uv/) if you haven't already.

Sync the project (creates a `.venv` and installs all dependencies):

```bash
uv sync
```


## Register the Jupyter kernelspec

Register the project's virtualenv as a Jupyter kernel so you can select it in JupyterLab:

```bash
uv run python -m ipykernel install --user --name ccmp-recipes --display-name "CCMP Recipes"
```

Then launch JupyterLab (from any environment that has it installed) and select the **CCMP Recipes** kernel when opening the notebook.

## Run

Open the notebook in JupyterLab and select the `ccmp-recipes` kernel:

```bash
uv run jupyter lab
```

Or run it non-interactively:

```bash
uv run jupyter execute virtualizarr_recipes.ipynb
```
