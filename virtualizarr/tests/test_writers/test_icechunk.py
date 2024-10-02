import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
from xarray import Dataset, open_dataset
from xarray.core.variable import Variable
from zarr import Array, Group, group

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.writers.icechunk import dataset_to_icechunk
from virtualizarr.zarr import ZArray

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
    def test_write_new_virtual_variable(
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

    def test_set_single_virtual_ref_without_encoding(
        self, icechunk_filestore: "IcechunkStore", simple_netcdf4: Path
    ):
        # TODO kerchunk doesn't work with zarr-python v3 yet so we can't use open_virtual_dataset and icechunk together!
        # vds = open_virtual_dataset(netcdf4_file, indexes={})

        # instead for now just write out byte ranges explicitly
        manifest = ChunkManifest(
            {"0.0": {"path": simple_netcdf4, "offset": 6144, "length": 48}}
        )
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
        vds = Dataset(
            {"foo": foo},
        )

        dataset_to_icechunk(vds, icechunk_filestore)

        root_group = group(store=icechunk_filestore)
        array = root_group["foo"]

        # check chunk references
        # TODO we can't explicitly check that the path/offset/length is correct because icechunk doesn't yet expose any get_virtual_refs method

        expected_ds = open_dataset(simple_netcdf4)
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(array, expected_array)

        # note: we don't need to test that committing works, because now we have confirmed
        # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.

    @pytest.mark.xfail(reason="Test doesn't account for scale factor encoding yet")
    def test_set_single_virtual_ref_with_encoding(
        self, icechunk_filestore: "IcechunkStore", netcdf4_file: Path
    ):
        # TODO kerchunk doesn't work with zarr-python v3 yet so we can't use open_virtual_dataset and icechunk together!
        # vds = open_virtual_dataset(netcdf4_file, indexes={})

        # instead for now just write out byte ranges explicitly
        manifest = ChunkManifest(
            {"0.0.0": {"path": netcdf4_file, "offset": 15419, "length": 7738000}}
        )
        zarray = ZArray(
            shape=(2920, 25, 53),
            chunks=(2920, 25, 53),
            dtype=np.dtype("int16"),
            compressor=None,
            filters=None,
            fill_value=None,
        )
        ma = ManifestArray(
            chunkmanifest=manifest,
            zarray=zarray,
        )
        air = Variable(data=ma, dims=["time", "lat", "lon"])
        vds = Dataset(
            {"air": air},
        )

        dataset_to_icechunk(vds, icechunk_filestore)

        root_group = group(store=icechunk_filestore)
        air_array = root_group["air"]
        print(air_array)

        # check chunk references
        # TODO we can't explicitly check that the path/offset/length is correct because icechunk doesn't yet expose any get_virtual_refs method

        expected_ds = open_dataset(netcdf4_file)
        expected_air_array = expected_ds["air"].to_numpy()
        npt.assert_equal(air_array, expected_air_array)

        # note: we don't need to test that committing works, because now we have confirmed
        # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


# TODO get test with encoding working

# TODO test writing grids of multiple chunks

# TODO test writing to a group that isn't the root group

# TODO test writing loadable variables

# TODO roundtripping tests - requires icechunk compatibility with xarray

# TODO test with S3 / minio
