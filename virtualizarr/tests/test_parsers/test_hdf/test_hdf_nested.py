from pathlib import Path

import h5py
import numpy as np
import zarr

from virtualizarr.tests.utils import manifest_store_from_hdf_url


def test_nested_h5(tmp_path: Path):
    h5_path = tmp_path / "my.h5"
    with h5py.File(h5_path, mode="w") as f:
        f["bar"] = np.arange(20)
        g = f.create_group("a_group")
        g["foo"] = np.arange(10)
    manifest_store = manifest_store_from_hdf_url(f"file://{h5_path}")
    z = zarr.open_group(manifest_store, mode="r", zarr_format=3)
    with h5py.File(h5_path, mode="r") as f:
        np.testing.assert_array_equal(f["bar"][...], z["bar"][...])
        np.testing.assert_array_equal(
            f["a_group"]["foo"][...], z["a_group"]["foo"][...]
        )
