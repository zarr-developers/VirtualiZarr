import contextlib
import pathlib

import fsspec
import fsspec.implementations.local
import fsspec.implementations.memory
import numpy as np
import pytest
import xarray as xr

from virtualizarr.manifests.utils import copy_and_replace_metadata
from virtualizarr.tests import requires_scipy
from virtualizarr.utils import _FsspecFSFromFilepath


@pytest.fixture
def dataset() -> xr.Dataset:
    return xr.Dataset(
        {"x": xr.DataArray([10, 20, 30], dims="a", coords={"a": [0, 1, 2]})}
    )


def test_fsspec_openfile_from_path(tmp_path: pathlib.Path, dataset: xr.Dataset) -> None:
    f = tmp_path / "dataset.nc"
    dataset.to_netcdf(f)

    result = _FsspecFSFromFilepath(filepath=f.as_posix()).open_file()
    assert isinstance(result, fsspec.implementations.local.LocalFileOpener)


@requires_scipy
def test_fsspec_openfile_memory(dataset: xr.Dataset):
    fs = fsspec.filesystem("memory")
    with contextlib.redirect_stderr(None):
        # Suppress "Exception ignored in: <function netcdf_file.close at ...>"
        with fs.open("dataset.nc", mode="wb") as f:
            dataset.to_netcdf(f, engine="h5netcdf")

    result = _FsspecFSFromFilepath(filepath="memory://dataset.nc").open_file()
    with result:
        assert isinstance(result, fsspec.implementations.memory.MemoryFile)


def test_copy_and_replace_metadata(array_v3_metadata):
    old_metadata = array_v3_metadata(
        shape=(10, 10),
        data_type=np.dtype("float32"),
        chunks=(5, 5),
        fill_value=0,
    )

    new_shape = (20, 20)
    new_chunks = (10, 10)

    # Test updating both shape and chunk shape
    updated_metadata = copy_and_replace_metadata(
        old_metadata, new_shape=new_shape, new_chunks=new_chunks
    )
    assert updated_metadata.shape == tuple(new_shape)
    assert updated_metadata.chunks == tuple(new_chunks)
    # Test other values are still the same
    assert updated_metadata.data_type == old_metadata.data_type
    assert updated_metadata.fill_value == old_metadata.fill_value
