# Installation

VirtualiZarr is available on PyPI via pip:

```shell
pip install virtualizarr
```

and on conda-forge:

```shell
conda install -c conda-forge virtualizarr
```

## Optional dependencies

VirtualiZarr has many optional dependencies, split into those for reading various file formats, and those for writing virtual references out to different formats.

Optional dependencies can be installed in groups via pip. For example to read HDF files and write virtual references to icechunk you could install all necessary dependencies via:

```shell
pip install "virtualizarr[hdf, icechunk]"
```

The full list of optional dependencies can be seen in the `pyproject.toml` file:

```{literalinclude} ../pyproject.toml
:start-at: "[project.optional-dependencies]"
:end-before: "test ="
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
