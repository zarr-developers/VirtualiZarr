import contextlib
import pathlib

import fsspec
import fsspec.implementations.local
import fsspec.implementations.memory
import pytest
import xarray as xr

from virtualizarr.utils import _fsspec_openfile_from_filepath


@pytest.fixture
def dataset() -> xr.Dataset:
    return xr.Dataset(
        {"x": xr.DataArray([10, 20, 30], dims="a", coords={"a": [0, 1, 2]})}
    )


def test_fsspec_openfile_from_path(tmp_path: pathlib.Path, dataset: xr.Dataset) -> None:
    f = tmp_path / "dataset.nc"
    dataset.to_netcdf(f)

    result = _fsspec_openfile_from_filepath(filepath=f.as_posix())
    assert isinstance(result, fsspec.implementations.local.LocalFile)


def test_fsspec_openfile_memory(dataset: xr.Dataset):
    fs = fsspec.filesystem("memory")
    with contextlib.redirect_stderr(None):
        # Suppress "Exception ignored in: <function netcdf_file.close at ...>"
        with fs.open("dataset.nc", mode="wb") as f:
            dataset.to_netcdf(f, engine="h5netcdf")

    result = _fsspec_openfile_from_filepath(filepath="memory://dataset.nc")
    with result:
        assert isinstance(result, fsspec.implementations.memory.MemoryFile)
