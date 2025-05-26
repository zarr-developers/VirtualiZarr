# Installation

VirtualiZarr is available on PyPI via pip:

```shell
pip install virtualizarr
```

and on conda-forge:

```shell
conda install -c conda-forge virtualizarr
```

Optional dependencies can be installed in groups via pip. For example to read HDF files and write virtual references to icechunk you could install all necessary dependencies via:

```shell
pip install "virtualizarr[hdf, icechunk]"
```

## Available optional dependency groups

VirtualiZarr has many optional dependencies, split into those for reading various file formats, and those for writing virtual references out to different formats. The available optional dependency groups are as follows:

```python exec="true"
import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

# For PEP 621 optional dependencies
print("```")
if "project" in data and "optional-dependencies" in data["project"]:
    for group, deps in data["project"]["optional-dependencies"].items():
        print(f"{group}:\n")
        for dep in deps:
            print(f"\t{dep}\n")
print("```")
```

The compound groups allow you to install multiple sets of dependencies at once, e.g. install every file reader via

```shell
pip install "virtualizarr[all_readers]"
```

The basic `pip install virtualizarr` will only install the minimal required dependencies, and so may not be particularly useful on its own.

## Install Test Dependencies

For local development you will want to install the test dependencies so that you can run all the tests in the test suite:

```shell
pip install '-e .[test]'
```

## Install Docs Dependencies

To build the documentation locally you will need further dependencies:

```shell
pip install '-e .[docs]'
```
