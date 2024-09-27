from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("icechunk")

from xarray import Dataset

from virtualizarr.writers.icechunk import dataset_to_icechunk

if TYPE_CHECKING:
    try:
        from icechunk import IcechunkStore
    except ImportError:
        IcechunkStore = Any


@pytest.fixture
async def icechunk_filestore(tmpdir) -> "IcechunkStore":
    from icechunk import IcechunkStore, StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))
    store = await IcechunkStore.open(storage=storage, mode="r+")

    return store


@pytest.mark.asyncio
async def test_write_to_icechunk(
    icechunk_filestore: "IcechunkStore", vds_with_manifest_arrays: Dataset
):
    dataset_to_icechunk(vds_with_manifest_arrays, icechunk_filestore)

    print(icechunk_filestore)
