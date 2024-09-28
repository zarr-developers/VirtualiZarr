import asyncio
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

from xarray import Dataset

from virtualizarr.writers.icechunk import dataset_to_icechunk

if TYPE_CHECKING:
    from icechunk import IcechunkStore


@pytest.fixture
def icechunk_filestore(tmpdir) -> "IcechunkStore":
    from icechunk import IcechunkStore, StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO if icechunk exposed a synchronous version of .open then we wouldn't need to use asyncio.run here
    store = asyncio.run(IcechunkStore.open(storage=storage, mode="r+"))

    # TODO instead yield store then store.close() ??
    return store


def test_write_to_icechunk(
    icechunk_filestore: "IcechunkStore", vds_with_manifest_arrays: Dataset
):
    dataset_to_icechunk(vds_with_manifest_arrays, icechunk_filestore)

    # TODO assert that arrays and references have been written


# TODO roundtripping tests - requires icechunk compatibility with xarray
