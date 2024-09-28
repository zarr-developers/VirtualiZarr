import asyncio
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

from xarray import Dataset
from zarr import group

from virtualizarr.writers.icechunk import dataset_to_icechunk

if TYPE_CHECKING:
    from icechunk import IcechunkStore


@pytest.fixture
def icechunk_filestore(tmpdir) -> "IcechunkStore":
    from icechunk import IcechunkStore, StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO if icechunk exposed a synchronous version of .open then we wouldn't need to use asyncio.run here
    # TODO is this the correct mode to use?
    store = asyncio.run(IcechunkStore.open(storage=storage, mode="r+"))

    # TODO instead yield store then store.close() ??
    return store


class TestWriteVirtualRefs:
    def test_write_new_variable(
        self, icechunk_filestore: "IcechunkStore", vds_with_manifest_arrays: Dataset
    ):
        dataset_to_icechunk(vds_with_manifest_arrays, icechunk_filestore)

        root_group = group(store=icechunk_filestore)
        assert root_group.attrs == {"something": 0}

        # TODO assert that arrays, array attrs, and references have been written

        # note: we don't need to test that committing actually works, because now we have confirmed
        # the refs are in the store (even uncommitted) it's icechunk's problem to manage now.


# TODO test writing loadable variables

# TODO roundtripping tests - requires icechunk compatibility with xarray
