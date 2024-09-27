import pytest

pytest.importorskip("icechunk")

from xarray import Dataset


from virtualizarr.writers.icechunk import dataset_to_icechunk


@pytest.mark.asyncio
async def test_write_to_icechunk(tmpdir, vds_with_manifest_arrays: Dataset):
    from icechunk import IcechunkStore, StorageConfig
    
    storage = StorageConfig.filesystem(str(tmpdir))
    store = await IcechunkStore.open(storage=storage, mode='r+')

    print(store)

    raise
    
    dataset_to_icechunk()
    ...
