from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import zarr

from virtualizarr.tests.utils import manifest_store_from_hdf_url


@pytest.fixture(scope="session")
def hdf5_nested_path(tmp_path_factory: pytest.TempPathFactory):
    h5_path = tmp_path_factory.getbasetemp() / "nested.h5"
    with h5py.File(h5_path, mode="w") as f:
        f["bar"] = np.arange(20)
        g = f.create_group("a_group")
        g["foo"] = np.arange(10)
    return h5_path


def test_nested_h5(hdf5_nested_path: Path):
    manifest_store = manifest_store_from_hdf_url(f"file://{hdf5_nested_path}")
    z = zarr.open_group(manifest_store, mode="r", zarr_format=3)
    with h5py.File(hdf5_nested_path, mode="r") as f:
        np.testing.assert_array_equal(f["bar"][...], cast(zarr.Array, z["bar"])[...])
        np.testing.assert_array_equal(
            f["a_group"]["foo"][...],
            cast(zarr.Array, cast(zarr.Group, z["a_group"])["foo"])[...],
        )


def test_nested_h5_fails_dataset_Creation(hdf5_nested_path: Path):
    manifest_store = manifest_store_from_hdf_url(f"file://{hdf5_nested_path}")
    with pytest.raises(NotImplementedError, match="Converting a ManifestStore"):
        manifest_store.to_virtual_dataset()
