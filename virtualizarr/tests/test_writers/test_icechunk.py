import asyncio
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

import numpy as np
from xarray import Dataset
from zarr import Array, Group, group

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
        vds = vds_with_manifest_arrays

        dataset_to_icechunk(vds, icechunk_filestore)

        # check attrs
        root_group = group(store=icechunk_filestore)
        assert isinstance(root_group, Group)
        assert root_group.attrs == {"something": 0}

        # TODO check against vds, then perhaps parametrize?

        # check array exists
        assert "a" in root_group
        arr = root_group["a"]
        assert isinstance(arr, Array)

        # check array metadata
        # TODO why doesn't a .zarr_format or .version attribute exist on zarr.Array?
        # assert arr.zarr_format == 3
        assert arr.shape == (2, 3)
        assert arr.chunks == (2, 3)
        assert arr.dtype == np.dtype("<i8")
        assert arr.order == "C"
        assert arr.fill_value == 0
        # TODO check compressor, filters?

        # check array attrs
        # TODO somehow this is broken by setting the dimension names???
        # assert dict(arr.attrs) == {"units": "km"}

        # check dimensions
        assert arr.attrs["DIMENSION_NAMES"] == ["x", "y"]

        # check chunk references

        # note: we don't need to test that committing actually works, because now we have confirmed
        # the refs are in the store (even uncommitted) it's icechunk's problem to manage now.


# TODO test writing loadable variables

# TODO roundtripping tests - requires icechunk compatibility with xarray
