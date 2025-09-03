import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
import xarray.testing as xrt
import zarr
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX
from virtualizarr.writers.icechunk import generate_chunk_key

icechunk = pytest.importorskip("icechunk")


if TYPE_CHECKING:
    from icechunk import (  # type: ignore[import-not-found]
        IcechunkStore,
        Repository,
        Storage,
    )


@pytest.fixture(scope="function")
def icechunk_storage(tmp_path: Path) -> "Storage":
    from icechunk import Storage

    return Storage.new_local_filesystem(str(tmp_path))


@pytest.fixture(scope="function")
def icechunk_repo(icechunk_storage: "Storage", tmp_path: Path) -> "Repository":
    config = icechunk.RepositoryConfig.default()

    container = icechunk.VirtualChunkContainer(
        url_prefix=PYTEST_TMP_DIRECTORY_URL_PREFIX,
        store=icechunk.local_filesystem_store(PYTEST_TMP_DIRECTORY_URL_PREFIX),
    )
    config.set_virtual_chunk_container(container)

    return icechunk.Repository.create(
        storage=icechunk_storage,
        config=config,
        authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
    )


@pytest.fixture(scope="function")
def icechunk_filestore(icechunk_repo: "Repository") -> "IcechunkStore":
    session = icechunk_repo.writable_session("main")
    return session.store


@pytest.mark.parametrize("kwarg", [("group", {}), ("append_dim", {})])
def test_invalid_kwarg_type(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    kwarg: tuple[str, Any],
):
    name, value = kwarg
    with pytest.raises(TypeError, match=name):
        vds_with_manifest_arrays.vz.to_icechunk(icechunk_filestore, **{name: value})


@pytest.mark.parametrize("group_path", [None, "", "/a", "a", "/a/b", "a/b", "a/b/"])
def test_write_new_virtual_variable(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    group_path: Optional[str],
):
    vds = vds_with_manifest_arrays

    vds.vz.to_icechunk(icechunk_filestore, group=group_path, validate_containers=False)

    # check attrs
    group = zarr.group(store=icechunk_filestore, path=group_path)
    assert isinstance(group, zarr.Group)
    assert group.attrs.asdict() == {"something": 0}

    # TODO check against vds, then perhaps parametrize?

    # check array exists
    assert "a" in group
    arr = group["a"]
    assert isinstance(arr, zarr.Array)

    # check array metadata
    assert arr.metadata.zarr_format == 3
    assert arr.shape == (2, 3)
    assert arr.chunks == (2, 3)
    assert arr.dtype == np.dtype("<i8")
    assert arr.order == "C"
    assert arr.fill_value == 0
    # TODO check compressor, filters?
    #

    # check array attrs
    assert dict(arr.attrs) == {"units": "km"}

    # check dimensions
    if isinstance(arr.metadata, ArrayV3Metadata):
        assert arr.metadata.dimension_names == ("x", "y")


def test_set_single_virtual_ref_without_encoding(
    icechunk_filestore: "IcechunkStore",
    icechunk_repo: "Repository",
    synthetic_vds,
):
    vds, arr = synthetic_vds
    vds = vds.drop_encoding()
    vds.vz.to_icechunk(icechunk_filestore)

    icechunk_filestore.session.commit("test")

    icechunk_readonly_session = icechunk_repo.readonly_session("main")
    with (
        xr.open_zarr(
            store=icechunk_readonly_session.store, zarr_format=3, consolidated=False
        ) as ds,
    ):
        np.testing.assert_equal(ds["foo"].data, arr)
    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


def test_set_single_virtual_ref_with_encoding(
    icechunk_filestore: "IcechunkStore",
    icechunk_repo: "Repository",
    synthetic_vds,
):
    vds, arr = synthetic_vds
    vds.vz.to_icechunk(icechunk_filestore)

    icechunk_filestore.session.commit("test")

    icechunk_readonly_session = icechunk_repo.readonly_session("main")
    with (
        xr.open_zarr(
            store=icechunk_readonly_session.store, zarr_format=3, consolidated=False
        ) as ds,
    ):
        # We wrote a numpy array to a file and added encoding={"scale_factor": 2} to the
        # metadata. So, we expect the array loaded by xarray to be twice the magnitude of
        # the original numpy array if writing and applying the encoding is working properly.
        np.testing.assert_equal(ds["foo"].data, arr * 2)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage
    # them now.


def test_set_grid_virtual_refs(icechunk_filestore: "IcechunkStore", synthetic_vds_grid):
    vds, arr = synthetic_vds_grid

    vds.vz.to_icechunk(icechunk_filestore)

    root_group = zarr.group(store=icechunk_filestore)
    observed = root_group["foo"]
    assert isinstance(observed, zarr.Array)

    npt.assert_equal(observed, arr)


def test_write_big_endian_value(icechunk_repo: "Repository", big_endian_synthetic_vds):
    vds, arr = big_endian_synthetic_vds
    vds = vds.drop_encoding()
    # Commit the first virtual dataset
    writable_session = icechunk_repo.writable_session("main")
    vds.vz.to_icechunk(writable_session.store)
    writable_session.commit("test commit")
    read_session = icechunk_repo.readonly_session(branch="main")
    with (
        xr.open_zarr(read_session.store, consolidated=False, zarr_format=3) as ds,
    ):
        np.testing.assert_equal(ds["foo"].data, arr)


def test_write_loadable_variable(
    icechunk_filestore: "IcechunkStore",
    simple_netcdf4: Path,
    array_v3_metadata,
):
    # instead for now just write out byte ranges explicitly
    manifest = ChunkManifest(
        {"0.0": {"path": str(simple_netcdf4), "offset": 6144, "length": 48}}
    )
    metadata = array_v3_metadata(
        shape=(3, 4),
        chunks=(3, 4),
        codecs=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )

    ma_v = xr.Variable(data=ma, dims=["x", "y"])

    la_v = xr.Variable(
        dims=["x", "y"],
        data=np.random.rand(3, 4),
        attrs={"units": "km"},
    )
    vds = xr.Dataset({"air": la_v}, {"pressure": ma_v})

    # Icechunk checksums currently store with second precision, so we need to make sure
    # the checksum_date is at least one second in the future
    checksum_date = datetime.now(timezone.utc) + timedelta(seconds=1)
    vds.vz.to_icechunk(icechunk_filestore, last_updated_at=checksum_date)

    root_group = zarr.group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, zarr.Array)
    assert air_array.shape == (3, 4)
    assert air_array.dtype == np.dtype("float64")
    assert air_array.attrs["units"] == "km"
    npt.assert_equal(air_array[:], la_v[:])

    pressure_array = root_group["pressure"]
    assert isinstance(pressure_array, zarr.Array)
    assert pressure_array.shape == (3, 4)
    assert pressure_array.dtype == np.dtype("int32")

    with xr.open_dataset(simple_netcdf4) as expected_ds:
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(pressure_array, expected_array)


def test_validate_containers(
    icechunk_filestore: "IcechunkStore",
    array_v3_metadata,
) -> None:
    # create some references referring to data that doesn't have a corresponding virtual chunk container
    manifest = ChunkManifest(
        {"0.0": {"path": "s3://bucket/path/file.nc", "offset": 0, "length": 100}}
    )
    metadata = array_v3_metadata(
        shape=(3, 4),
        chunks=(3, 4),
        codecs=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    vds = xr.Dataset(
        {
            "foo": (["x", "y"], ma),
            # include some non-virtual data too
            "bar": (["x", "y"], np.ones((3, 4))),
        },
    )

    # assert that an error is raised when attempting to write to icechunk
    with pytest.raises(
        ValueError, match="No Virtual Chunk Container set which supports prefix"
    ):
        vds.vz.to_icechunk(icechunk_filestore)

    # assert that no uncommitted changes have been written to Icechunk session
    # Idea is that session has not been "polluted" with half-written changes
    session = icechunk_filestore.session
    # TODO could use https://github.com/earth-mover/icechunk/issues/1165 if it gets implemented
    assert not session.has_uncommitted_changes, session.status()


@pytest.fixture(scope="function")
def icechunk_repo_no_chunk_container(tmp_path: Path) -> "Repository":
    icechunk_storage = icechunk.Storage.new_local_filesystem(
        str(tmp_path) + "icechunk_1"
    )
    config = icechunk.RepositoryConfig.default()

    return icechunk.Repository.create(
        storage=icechunk_storage,
        config=config,
        # TODO do we need this?
        authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
    )


# TODO test with zero virtual chunk containers
def test_raise_if_zero_chunk_containers(
    icechunk_repo_no_chunk_container: "Repository",
    array_v3_metadata,
):
    # create some references referring to data that doesn't have a corresponding virtual chunk container
    manifest = ChunkManifest(
        {"0.0": {"path": "s3://bucket/path/file.nc", "offset": 0, "length": 100}}
    )
    metadata = array_v3_metadata(
        shape=(3, 4),
        chunks=(3, 4),
        codecs=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    vds = xr.Dataset(
        {
            "foo": (["x", "y"], ma),
            # include some non-virtual data too
            "bar": (["x", "y"], np.ones((3, 4))),
        },
    )

    session = icechunk_repo_no_chunk_container.writable_session("main")

    # assert that an error is raised when attempting to write to icechunk
    with pytest.raises(ValueError, match="No Virtual Chunk Containers set"):
        vds.vz.to_icechunk(session.store)

    # assert that no uncommitted changes have been written to Icechunk session
    # Idea is that session has not been "polluted" with half-written changes
    # TODO could use https://github.com/earth-mover/icechunk/issues/1165 if it gets implemented
    assert not session.has_uncommitted_changes, session.status()


def test_checksum(
    icechunk_filestore: "IcechunkStore",
    tmpdir: Path,
    array_v3_metadata,
):
    from icechunk import IcechunkError

    netcdf_path = tmpdir / "test.nc"
    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4) * 2
    var = xr.Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    ds.to_netcdf(netcdf_path)

    # instead for now just write out byte ranges explicitly
    manifest = ChunkManifest(
        {"0.0": {"path": str(netcdf_path), "offset": 6144, "length": 48}}
    )
    metadata = array_v3_metadata(
        shape=(3, 4),
        chunks=(3, 4),
        codecs=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )

    ma_v = xr.Variable(data=ma, dims=["x", "y"])

    vds = xr.Dataset({"pressure": ma_v})

    # default behaviour is to create a checksum based on the current time
    vds.vz.to_icechunk(icechunk_filestore)

    # Make sure the checksum_date is at least one second in the past before trying to overwrite referenced file with new data
    # This represents someone coming back much later and overwriting archival data
    time.sleep(1)

    # Fail if anything but None or a datetime is passed to last_updated_at
    with pytest.raises(TypeError):
        vds.vz.to_icechunk(icechunk_filestore, last_updated_at="not a datetime")  # type: ignore

    root_group = zarr.group(store=icechunk_filestore)
    pressure_array = root_group["pressure"]
    assert isinstance(pressure_array, zarr.Array)
    assert pressure_array.shape == (3, 4)
    assert pressure_array.dtype == np.dtype("int32")

    with xr.open_dataset(netcdf_path) as expected_ds:
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(pressure_array, expected_array)

    # Now we can overwrite the simple_netcdf4 file with new data to make sure that
    # the checksum_date is being used to determine if the data is valid
    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4) * 2
    var = xr.Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    ds.to_netcdf(netcdf_path)

    # TODO assert that icechunk knows the correct last_updated_at for this chunk
    # TODO ideally use icechunk's get_chunk_ref to directly interrogate the last_updated_time
    # however this is currently only available in rust

    # Now if we try to read the data back in, it should fail because the checksum_date
    # is newer than the last_updated_at
    with pytest.raises(IcechunkError):
        pressure_array = root_group["pressure"]
        assert isinstance(pressure_array, zarr.Array)
        npt.assert_equal(pressure_array, arr)


def test_generate_chunk_key_no_offset():
    # Test case without any offset (append_axis and existing_num_chunks are None)
    index = (1, 2, 3)
    result = generate_chunk_key(index)
    assert result == [1, 2, 3], (
        "The chunk key should match the index without any offset."
    )


def test_generate_chunk_key_with_offset():
    # Test case with offset on append_axis 1
    index = (1, 2, 3)
    append_axis = 1
    existing_num_chunks = 5
    result = generate_chunk_key(
        index, append_axis=append_axis, existing_num_chunks=existing_num_chunks
    )
    assert result == [1, 7, 3], "The chunk key should offset the second index by 5."


def test_generate_chunk_key_zero_offset():
    # Test case where existing_num_chunks is 0 (no offset should be applied)
    index = (4, 5, 6)
    append_axis = 1
    existing_num_chunks = 0
    result = generate_chunk_key(
        index, append_axis=append_axis, existing_num_chunks=existing_num_chunks
    )
    assert result == [4, 5, 6], (
        "No offset should be applied when existing_num_chunks is 0."
    )


def test_generate_chunk_key_append_axis_out_of_bounds():
    # Edge case where append_axis is out of bounds
    index = (3, 4)
    append_axis = 2  # This is out of bounds for a 2D index
    with pytest.raises(ValueError):
        generate_chunk_key(index, append_axis=append_axis, existing_num_chunks=1)


def test_roundtrip_coords(
    manifest_array, icechunk_filestore: "IcechunkStore", icechunk_repo: "Repository"
):
    # regression test for GH issue #574

    vds = xr.Dataset(
        data_vars={
            "data": (
                ["x", "y", "t"],
                manifest_array(shape=(4, 2, 3), chunks=(2, 1, 1)),
            ),
        },
        coords={
            "coord_3d": (
                ["x", "y", "t"],
                manifest_array(shape=(4, 2, 3), chunks=(2, 1, 1)),
            ),
            "coord_2d": (["x", "y"], manifest_array(shape=(4, 2), chunks=(2, 1))),
            "coord_1d": (["t"], manifest_array(shape=(3,), chunks=(1,))),
            "coord_0d": ([], manifest_array(shape=(), chunks=())),
        },
    )
    vds.vz.to_icechunk(icechunk_filestore, validate_containers=False)
    icechunk_filestore.session.commit("test")

    icechunk_readonly_session = icechunk_repo.readonly_session("main")
    roundtrip = xr.open_zarr(icechunk_readonly_session.store, consolidated=False)
    assert set(roundtrip.coords) == set(vds.coords)


class TestWarnIfNotVirtual:
    def test_warn_if_no_virtual_vars_dataset(self, icechunk_filestore: "IcechunkStore"):
        non_virtual_ds = xr.Dataset({"foo": ("x", [10, 20, 30]), "x": ("x", [1, 2, 3])})
        with pytest.warns(UserWarning, match="non-virtual"):
            non_virtual_ds.vz.to_icechunk(icechunk_filestore)

    def test_warn_if_no_virtual_vars_datatree(
        self, icechunk_filestore: "IcechunkStore"
    ):
        non_virtual_ds = xr.Dataset({"foo": ("x", [10, 20, 30]), "x": ("x", [1, 2, 3])})
        non_virtual_dt = xr.DataTree.from_dict(
            {"/": non_virtual_ds, "/group": non_virtual_ds}
        )
        with pytest.warns(UserWarning, match="non-virtual"):
            non_virtual_dt.vz.to_icechunk(icechunk_filestore)


class TestAppend:
    """
    Tests for appending to existing icechunk store.
    """

    # Success cases
    ## When appending to a single virtual ref without encoding, it succeeds
    def test_append_virtual_ref_without_encoding(
        self, icechunk_repo: "Repository", synthetic_vds
    ):
        vds, arr = synthetic_vds
        vds = vds.drop_encoding()
        # Commit the first virtual dataset
        writable_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(writable_session.store)
        writable_session.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        append_session = icechunk_repo.writable_session("main")

        # Append the same dataset to the same store
        vds.vz.to_icechunk(append_session.store, append_dim="x")
        append_session.commit("appended data")

        second_append_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(second_append_session.store, append_dim="x")
        second_append_session.commit("appended data again")

        read_session = icechunk_repo.readonly_session(branch="main")
        with (
            xr.open_zarr(read_session.store, consolidated=False, zarr_format=3) as ds,
        ):
            np.testing.assert_equal(
                ds["foo"].data, np.concatenate([arr, arr, arr], axis=1)
            )

    def test_append_virtual_ref_with_encoding(
        self, icechunk_repo: "Repository", synthetic_vds
    ):
        vds, arr = synthetic_vds
        # Commit the first virtual dataset
        writable_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(writable_session.store)
        writable_session.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        append_session = icechunk_repo.writable_session("main")

        # Append the same dataset to the same store
        vds.vz.to_icechunk(append_session.store, append_dim="x")
        append_session.commit("appended data")

        second_append_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(second_append_session.store, append_dim="x")
        second_append_session.commit("appended data again")

        read_session = icechunk_repo.readonly_session(branch="main")
        with (
            xr.open_zarr(read_session.store, consolidated=False, zarr_format=3) as ds,
        ):
            np.testing.assert_equal(
                ds["foo"].data,
                np.concatenate([arr, arr, arr], axis=1) * 2,
            )

    ## When appending to a virtual ref with encoding, it succeeds
    @pytest.mark.asyncio
    async def test_append_with_multiple_root_arrays(
        self, icechunk_repo: "Repository", synthetic_vds_multiple_vars
    ):
        vds, arr = synthetic_vds_multiple_vars
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Append the same dataset to the same store
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(icechunk_filestore_append.store, append_dim="x")
        icechunk_filestore_append.commit("appended data")

        read_session = icechunk_repo.readonly_session(branch="main")
        with (
            xr.open_zarr(read_session.store, consolidated=False, zarr_format=3) as ds,
        ):
            np.testing.assert_equal(
                ds["foo"].data, np.concatenate([arr, arr], axis=1) * 2
            )
            np.testing.assert_equal(
                ds["bar"].data, np.concatenate([arr, arr], axis=1) * 2
            )

    # When appending to a virtual ref with compression, it succeeds
    def test_append_with_compression_succeeds(
        self,
        icechunk_repo: "Repository",
        netcdf4_files_factory: Callable,
        compressed_synthetic_vds,
    ):
        vds, arr = compressed_synthetic_vds
        # Commit the first virtual dataset
        writable_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(writable_session.store)
        writable_session.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        append_session = icechunk_repo.writable_session("main")

        # Append the same dataset to the same store
        vds.vz.to_icechunk(append_session.store, append_dim="x")
        append_session.commit("appended data")

        second_append_session = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(second_append_session.store, append_dim="x")
        second_append_session.commit("appended data again")

        read_session = icechunk_repo.readonly_session(branch="main")
        with (
            xr.open_zarr(read_session.store, consolidated=False, zarr_format=3) as ds,
        ):
            np.testing.assert_equal(
                ds["foo"].data,
                np.concatenate([arr, arr, arr], axis=1),
            )

    ## When chunk shapes are different it fails
    def test_append_with_different_chunking_fails(
        self,
        icechunk_repo: "Repository",
        simple_netcdf4: str,
        virtual_dataset: Callable,
    ):
        # Generate a virtual dataset with specific chunking
        vds = virtual_dataset(url=simple_netcdf4, chunk_shape=(3, 4))

        # Commit the dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Try to append dataset with different chunking, expect failure
        vds_different_chunking = virtual_dataset(url=simple_netcdf4, chunk_shape=(1, 1))
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(
            ValueError, match="Cannot concatenate arrays with inconsistent chunk shapes"
        ):
            vds_different_chunking.vz.to_icechunk(
                icechunk_filestore_append.store, append_dim="x"
            )

    ## When encoding is different it fails
    def test_append_with_different_encoding_fails(
        self,
        icechunk_repo: "Repository",
        simple_netcdf4: str,
        virtual_dataset: Callable,
    ):
        # Generate datasets with different encoding
        vds1 = virtual_dataset(url=simple_netcdf4, encoding={"scale_factor": 0.1})
        vds2 = virtual_dataset(url=simple_netcdf4, encoding={"scale_factor": 0.01})

        # Commit the first dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Try to append with different encoding, expect failure
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(
            ValueError,
            match="Cannot concatenate arrays with different values for encoding",
        ):
            vds2.vz.to_icechunk(icechunk_filestore_append.store, append_dim="x")

    def test_dimensions_do_not_align(
        self,
        icechunk_repo: "Repository",
        simple_netcdf4: str,
        virtual_dataset: Callable,
    ):
        # Generate datasets with different lengths on the non-append dimension (x)
        vds1 = virtual_dataset(
            # {'x': 5, 'y': 4}
            url=simple_netcdf4,
            shape=(5, 4),
        )
        vds2 = virtual_dataset(
            # {'x': 6, 'y': 4}
            url=simple_netcdf4,
            shape=(6, 4),
        )

        # Commit the first dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Attempt to append dataset with different length in non-append dimension, expect failure
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(ValueError, match="Cannot concatenate arrays with shapes"):
            vds2.vz.to_icechunk(icechunk_filestore_append.store, append_dim="y")

    def test_append_dim_not_in_dims_raises_error(
        self,
        icechunk_repo: "Repository",
        simple_netcdf4: str,
        virtual_dataset: Callable,
    ):
        """
        Test that attempting to append with an append_dim not present in dims raises a ValueError.
        """
        vds = virtual_dataset(
            url=simple_netcdf4, shape=(5, 4), chunk_shape=(5, 4), dims=["x", "y"]
        )

        icechunk_filestore = icechunk_repo.writable_session("main")
        vds.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("initial commit")

        # Attempt to append using a non-existent append_dim "z"
        icechunk_filestore_append = icechunk_repo.writable_session("main")

        with pytest.raises(
            ValueError,
            match="append_dim 'z' does not match any existing dataset dimensions",
        ):
            vds.vz.to_icechunk(icechunk_filestore_append.store, append_dim="z")


# TODO test with S3 / minio


def test_write_empty_chunk(
    icechunk_filestore: "IcechunkStore",
    array_v3_metadata,
):
    # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/740

    # ManifestArray containing empty chunk
    manifest = ChunkManifest({"0": {"path": "", "offset": 0, "length": 0}})
    metadata = array_v3_metadata(
        shape=(5,),
        chunks=(5,),
        data_type=np.dtype("int32"),
        fill_value=10,
    )
    marr = ManifestArray(chunkmanifest=manifest, metadata=metadata)
    vds = xr.Dataset({"a": ("x", marr)})

    # empty chunks should never be written
    vds.vz.to_icechunk(icechunk_filestore)

    # when opened they should be treated as fill_value
    roundtrip = xr.open_zarr(
        icechunk_filestore, zarr_format=3, consolidated=False, chunks={}
    )
    expected_values = np.full(shape=(5,), fill_value=10, dtype=np.dtype("int32"))
    expected = xr.Variable(data=expected_values, dims=["x"])
    xrt.assert_identical(roundtrip["a"].variable, expected)
