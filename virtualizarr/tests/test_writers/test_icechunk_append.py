from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
from xarray import Dataset, open_dataset
from xarray.core.variable import Variable
from zarr import group  # type: ignore[import-untyped]

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.writers.icechunk import dataset_to_icechunk
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    from icechunk import StorageConfig  # type: ignore[import-not-found]


@pytest.fixture(scope="function")
def icechunk_storage(tmpdir) -> "StorageConfig":
    from icechunk import StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO instead yield store then store.close() ??
    return storage


def gen_virtual_dataset(file_uri: str):
    manifest = ChunkManifest({"0.0": {"path": file_uri, "offset": 6144, "length": 48}})
    zarray = ZArray(
        shape=(3, 4),
        chunks=(3, 4),
        dtype=np.dtype("int32"),
        compressor=None,
        filters=None,
        fill_value=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        zarray=zarray,
    )
    foo = Variable(data=ma, dims=["x", "y"])
    return Dataset(
        {"foo": foo},
    )


def test_set_single_virtual_ref_without_encoding(
    icechunk_storage: "StorageConfig", simple_netcdf4: str
):
    import xarray as xr
    from icechunk import IcechunkStore

    # generate virtual dataset
    vds = gen_virtual_dataset(simple_netcdf4)

    # create the icechunk store and commit the first virtual dataset
    icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
    dataset_to_icechunk(vds, icechunk_filestore)
    icechunk_filestore.commit(
        "test commit"
    )  # need to commit it in order to append to it in the next lines

    # Append the same dataset to the same store
    icechunk_filestore_append = IcechunkStore.open_existing(
        storage=icechunk_storage, mode="a"
    )
    dataset_to_icechunk(vds, icechunk_filestore_append, append_dim="x")

    root_group = group(store=icechunk_filestore_append)
    array = root_group["foo"]

    expected_ds = open_dataset(simple_netcdf4)
    expected_array = xr.concat(
        [expected_ds["foo"], expected_ds["foo"]], dim="x"
    ).to_numpy()
    npt.assert_equal(array, expected_array)
