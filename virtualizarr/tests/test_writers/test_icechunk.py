from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

from xarray import Dataset

from virtualizarr.writers.icechunk import dataset_to_icechunk

if TYPE_CHECKING:
    from icechunk import IcechunkStore


@pytest.fixture
async def icechunk_filestore(tmpdir) -> "IcechunkStore":
    from icechunk import IcechunkStore, StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))
    # TODO if I use asyncio.run can I avoid this fixture and tests being async functions?
    store = await IcechunkStore.open(storage=storage, mode="r+")

    # TODO instead yield store then store.close() ??
    return store


@pytest.mark.asyncio
async def test_write_to_icechunk(
    icechunk_filestore: "IcechunkStore", vds_with_manifest_arrays: Dataset
):
    store = await icechunk_filestore

    dataset_to_icechunk(vds_with_manifest_arrays, store)

    # TODO assert that arrays and references have been written


# TODO roundtripping tests - requires icechunk compatibility with xarray
