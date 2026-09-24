import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import numpy.testing as npt
import obstore as obs
import pandas as pd
import pytest
import xarray as xr
import xarray.testing as xrt
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from zarr.codecs import BytesCodec
from zarr.core.metadata import ArrayV3Metadata
from zarr.dtype import parse_data_type
from zarr.errors import ContainsGroupError

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.parsers import HDFParser
from virtualizarr.parsers.zarr import ZarrParser
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX

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


@pytest.mark.parametrize("kwarg", [("group", {}), ("mode", {}), ("append_dim", {})])
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


@pytest.mark.parametrize("mode", [None, "w-"])
def test_write_to_existing_group_fails_by_default(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    mode: Optional[str],
):
    vds = vds_with_manifest_arrays
    vds.vz.to_icechunk(icechunk_filestore, validate_containers=False)

    with pytest.raises(ContainsGroupError):
        vds.vz.to_icechunk(icechunk_filestore, mode=mode, validate_containers=False)


def test_write_variables_across_commits_with_mode_a(
    icechunk_repo: "Repository",
    synthetic_vds_multiple_vars,
):
    # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/1001
    vds, arr = synthetic_vds_multiple_vars

    session1 = icechunk_repo.writable_session("main")
    vds[["foo"]].vz.to_icechunk(session1.store)
    session1.commit("wrote foo")

    session2 = icechunk_repo.writable_session("main")
    vds[["bar"]].vz.to_icechunk(session2.store, mode="a")
    session2.commit("wrote bar")

    with xr.open_zarr(
        icechunk_repo.readonly_session("main").store, zarr_format=3, consolidated=False
    ) as ds:
        # both variables have encoding={"scale_factor": 2}
        np.testing.assert_equal(ds["foo"].data, arr * 2)
        np.testing.assert_equal(ds["bar"].data, arr * 2)


def test_write_parent_group_after_child_group_with_mode_a(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
):
    # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/1001
    vds = vds_with_manifest_arrays
    vds.vz.to_icechunk(
        icechunk_filestore, group="supgroup/subgroup", validate_containers=False
    )
    vds.vz.to_icechunk(
        icechunk_filestore, group="supgroup", mode="a", validate_containers=False
    )

    assert "a" in zarr.group(store=icechunk_filestore, path="supgroup")
    assert "a" in zarr.group(store=icechunk_filestore, path="supgroup/subgroup")


def test_mode_w_overwrites_existing_group(
    icechunk_filestore: "IcechunkStore",
    synthetic_vds_multiple_vars,
):
    vds, arr = synthetic_vds_multiple_vars
    vds.vz.to_icechunk(icechunk_filestore)
    vds[["foo"]].vz.to_icechunk(icechunk_filestore, mode="w")

    group = zarr.group(store=icechunk_filestore)
    assert "foo" in group
    assert "bar" not in group


def test_invalid_mode(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
):
    with pytest.raises(ValueError, match="mode"):
        vds_with_manifest_arrays.vz.to_icechunk(icechunk_filestore, mode="r+")


@pytest.mark.parametrize("mode", ["w", "w-"])
def test_mode_incompatible_with_append_dim(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    mode: str,
):
    with pytest.raises(ValueError, match="append_dim or region"):
        vds_with_manifest_arrays.vz.to_icechunk(
            icechunk_filestore, mode=mode, append_dim="x"
        )


def test_write_datatree_to_existing_groups_with_mode_a(
    icechunk_repo: "Repository",
    synthetic_vds_multiple_vars,
):
    vds, arr = synthetic_vds_multiple_vars

    session1 = icechunk_repo.writable_session("main")
    vdt1 = xr.DataTree.from_dict({"nested/group": vds[["foo"]]})
    vdt1.vz.to_icechunk(session1.store)
    session1.commit("wrote foo")

    session2 = icechunk_repo.writable_session("main")
    vdt2 = xr.DataTree.from_dict({"nested/group": vds[["bar"]]})
    vdt2.vz.to_icechunk(session2.store, mode="a")
    session2.commit("wrote bar")

    with xr.open_zarr(
        icechunk_repo.readonly_session("main").store,
        zarr_format=3,
        consolidated=False,
        group="nested/group",
    ) as ds:
        np.testing.assert_equal(ds["foo"].data, arr * 2)
        np.testing.assert_equal(ds["bar"].data, arr * 2)


def test_mode_a_raises_when_an_existing_array_has_different_codecs(
    icechunk_filestore: "IcechunkStore", synthetic_vds, compressed_synthetic_vds
):
    synthetic_vds[0].vz.to_icechunk(icechunk_filestore)

    with pytest.raises(ValueError, match="with different codecs\\."):
        compressed_synthetic_vds[0].vz.to_icechunk(icechunk_filestore, mode="a")


def test_mode_a_raises_when_an_existing_array_has_different_dimension_names(
    icechunk_filestore: "IcechunkStore", synthetic_vds
):
    vds, _ = synthetic_vds
    vds.vz.to_icechunk(icechunk_filestore)

    with pytest.raises(ValueError, match="with different dimension_names\\."):
        vds.rename({"x": "t"}).vz.to_icechunk(icechunk_filestore, mode="a")


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


def test_set_inlined_and_virtual_refs(
    icechunk_filestore: "IcechunkStore",
    icechunk_repo: "Repository",
    tmp_path: Path,
):
    # ManifestArray with shape (2, 2), chunks (1, 2): position (0, .) is inlined
    # with values [1, 2]; position (1, .) is virtual with values [3, 4] read from
    # a file in the tmp dir (covered by the icechunk_repo virtual chunk container).
    inlined_arr = np.array([[1, 2]], dtype="<i4")
    virtual_arr = np.array([[3, 4]], dtype="<i4")
    inlined_bytes = inlined_arr.tobytes()
    virtual_bytes = virtual_arr.tobytes()

    filepath = str(tmp_path / "data_chunk")
    obs.put(obs.store.LocalStore(), filepath, virtual_bytes)

    manifest = ChunkManifest(
        entries={
            "0.0": {
                "path": "",
                "offset": 0,
                "length": len(inlined_bytes),
                "data": inlined_bytes,
            },
            "1.0": {
                "path": filepath,
                "offset": 0,
                "length": len(virtual_bytes),
            },
        }
    )
    metadata = ArrayV3Metadata(
        shape=(2, 2),
        data_type=parse_data_type(np.dtype("<i4"), zarr_format=3),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": (1, 2)}},
        chunk_key_encoding={"name": "default"},
        fill_value=0,
        codecs=[BytesCodec()],
        attributes={},
        dimension_names=("y", "x"),
        storage_transformers=None,
    )
    ma = ManifestArray(chunkmanifest=manifest, metadata=metadata)
    vds = xr.Dataset({"foo": xr.Variable(data=ma, dims=["y", "x"])})

    vds.vz.to_icechunk(icechunk_filestore)
    icechunk_filestore.session.commit("test")

    icechunk_readonly_session = icechunk_repo.readonly_session("main")
    with xr.open_zarr(
        store=icechunk_readonly_session.store, zarr_format=3, consolidated=False
    ) as ds:
        np.testing.assert_equal(ds["foo"].data, np.array([[1, 2], [3, 4]], dtype="<i4"))


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
    def test_append_with_multiple_root_arrays(
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


class TestRegion:
    @pytest.fixture()
    def combined_synthetic_vds(self, synthetic_vds):
        vds, arr = synthetic_vds
        vds = vds.drop_encoding()
        vds_grid = [
            [
                vds.assign_coords(
                    y=range(chunk_y * len(vds.y), (chunk_y + 1) * len(vds.y)),
                    x=range(chunk_x * len(vds.x), (chunk_x + 1) * len(vds.x)),
                )
                for chunk_x in range(3)
            ]
            for chunk_y in range(3)
        ]
        combined_empty_vds = xr.combine_nested(vds_grid, concat_dim=["y", "x"])
        # delete all references, this is so we can just write the coordinates
        # and zarr metadata to the store. This is typically done instead with
        # dask arrays and "compute=False", but we better not depend on dask just for this
        combined_empty_vds["foo"].data = ManifestArray(
            metadata=combined_empty_vds["foo"].data.metadata,
            chunkmanifest=ChunkManifest({}, shape=combined_empty_vds["foo"].data.shape),
        )
        return vds, arr, vds_grid, combined_empty_vds

    @pytest.fixture(scope="function")
    def initialized_repo(self, icechunk_repo: "Repository", combined_synthetic_vds):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        # initialize empty
        init_session = icechunk_repo.writable_session("main")
        combined_empty_vds.vz.to_icechunk(init_session.store)
        init_session.commit("test commit")
        return icechunk_repo

    def test_initialized_repo_is_empty(self, initialized_repo):
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            assert np.all(ds["foo"].data == 0)  # fill_value is 0

    def test_write_region_auto(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][1]
        write_session = initialized_repo.writable_session("main")
        my_vds.vz.to_icechunk(write_session.store, region="auto")
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_explicit_auto(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][1]
        write_session = initialized_repo.writable_session("main")
        my_vds.vz.to_icechunk(write_session.store, region={"y": "auto", "x": "auto"})
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_explicit(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][1]
        write_session = initialized_repo.writable_session("main")
        my_vds.vz.to_icechunk(
            write_session.store,
            region={
                "y": slice(len(vds["y"]), 2 * len(vds["y"])),
                "x": slice(len(vds["x"]), 2 * len(vds["x"])),
            },
        )
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_multiple(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = xr.combine_nested(vds_grid[1], concat_dim=["x"])
        write_session = initialized_repo.writable_session("main")
        my_vds.vz.to_icechunk(write_session.store, region="auto")
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_writes_whole_unspecified_dim(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        # the virtual dataset spans all of dimension 'x', for which it doesn't have coordinates
        # and we want to write to a region in 'y'
        my_vds = xr.combine_nested(vds_grid[1], concat_dim=["x"])
        my_vds = my_vds.drop_vars("x")
        write_session = initialized_repo.writable_session("main")
        my_vds.vz.to_icechunk(write_session.store, region={"y": "auto"})
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            actual = ds["foo"] != 0
            expected = (
                ds["y"].isin(my_vds["y"]).broadcast_like(actual).transpose(*actual.dims)
            )
            xrt.assert_allclose(expected, actual)

    def test_write_region_auto_without_dimension(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        """
        When the coordinate for an "auto" dimension is missing, xarray
        assumes that we want to write starting at index 0, even
        if the written dataset doesn't span the whole dimension.
        So VirtualiZarr behaves consistently with xarray here.
        """
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][0]
        my_vds_to_insert = my_vds.drop_vars("x")
        write_session = initialized_repo.writable_session("main")
        my_vds_to_insert.vz.to_icechunk(write_session.store, region="auto")
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_explicit_auto_without_dimension(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        """
        When the coordinate for an "auto" dimension is missing, xarray
        assumes that we want to write starting at index 0, even
        if the written dataset doesn't span the whole dimension.
        So VirtualiZarr behaves consistently with xarray here.
        """
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][0]
        my_vds_to_insert = my_vds.drop_vars("x")
        write_session = initialized_repo.writable_session("main")
        my_vds_to_insert.vz.to_icechunk(
            write_session.store, region={"y": "auto", "x": "auto"}
        )
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                initialized_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )

    def test_write_region_unaligned_chunks_raises(
        self,
        initialized_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        my_vds = vds_grid[1][1]
        my_vds = my_vds.assign_coords(x=my_vds["x"] - 1)
        write_session = initialized_repo.writable_session("main")
        with pytest.raises(ValueError, match="is not aligned to whole chunks"):
            my_vds.vz.to_icechunk(
                write_session.store, region={"y": "auto", "x": "auto"}
            )

    def test_write_datatree_region(
        self,
        icechunk_repo: "Repository",
        combined_synthetic_vds,
    ):
        vds, arr, vds_grid, combined_empty_vds = combined_synthetic_vds

        combined_empty_vdt = xr.DataTree.from_dict({"nested/group": combined_empty_vds})
        init_session = icechunk_repo.writable_session("main")
        combined_empty_vdt.vz.to_icechunk(init_session.store)
        init_session.commit("init repo")

        my_vds = vds_grid[1][1]
        write_session = icechunk_repo.writable_session("main")
        my_vdt = xr.DataTree.from_dict({"nested/group": my_vds})
        my_vdt.vz.to_icechunk(write_session.store, region="auto")
        write_session.commit("test commit")

        # check that references are written
        with (
            xr.open_zarr(
                icechunk_repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
                group="nested/group",
            ) as ds,
        ):
            is_written = ds["y"].isin(my_vds["y"]) & ds["x"].isin(my_vds["x"])
            xrt.assert_allclose(
                is_written,
                ds["foo"] != 0,
            )


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


def test_sharded_array_roundtrip_icechunk(icechunk_repo, tmp_path):
    """
    Test that a sharded Zarr V3 array preserves shard and chunk shapes through icechunk.

    Regression test for https://github.com/zarr-developers/VirtualiZarr/issues/951.
    """
    filepath = str(tmp_path / "test_sharded.zarr")

    # Create a sharded zarr store
    data = np.arange(12 * 12, dtype="float32").reshape(12, 12)
    ds = xr.Dataset({"data": (("x", "y"), data)})
    ds.to_zarr(
        filepath,
        encoding={"data": {"chunks": (3, 3), "shards": (6, 6)}},
        consolidated=False,
        zarr_format=3,
    )

    # Verify original shapes
    original_arr = zarr.open_array(filepath + "/data", mode="r")
    assert original_arr.shards == (6, 6)
    assert original_arr.chunks == (3, 3)

    # Virtualize with ZarrParser
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()
    vds = open_virtual_dataset(url=filepath, registry=registry, parser=parser)

    # Write to icechunk
    session = icechunk_repo.writable_session("main")
    vds.vz.to_icechunk(session.store, validate_containers=False)
    session.commit("test")

    # Read back from icechunk and verify shapes are preserved
    ro_session = icechunk_repo.readonly_session("main")
    ic_arr = zarr.open_array(ro_session.store, path="data", mode="r")
    assert ic_arr.shards == (6, 6), f"Expected shard shape (6,6), got {ic_arr.shards}"
    assert ic_arr.chunks == (3, 3), f"Expected chunk shape (3,3), got {ic_arr.chunks}"

    # Verify data values match
    ic_ds = xr.open_zarr(ro_session.store, zarr_format=3, consolidated=False)
    npt.assert_array_equal(ic_ds["data"].values, data)


def test_concat_sharded_arrays_along_new_dim_roundtrip_icechunk(
    icechunk_repo, tmp_path
):
    """
    Concatenating sharded virtual arrays along a *new* dimension must produce readable
    data: the shard's inner chunk_shape has to gain the same length-1 axis, and doing so
    leaves the shard's inner chunk grid, index layout and chunk bytes untouched.

    Regression test for https://github.com/zarr-developers/VirtualiZarr/issues/1076.
    """
    datasets, vdss = [], []
    for i in range(2):
        filepath = str(tmp_path / f"step_{i}.zarr")
        data = np.arange(i * 144, (i + 1) * 144, dtype="float32").reshape(12, 12)
        ds = xr.Dataset({"data": (("x", "y"), data)})
        ds.to_zarr(
            filepath,
            encoding={"data": {"chunks": (3, 3), "shards": (12, 12)}},
            consolidated=False,
            zarr_format=3,
        )
        datasets.append(ds)

        registry = ObjectStoreRegistry(
            {f"file://{filepath}": LocalStore(prefix=filepath)}
        )
        vdss.append(
            open_virtual_dataset(url=filepath, registry=registry, parser=ZarrParser())
        )

    # this is what used to raise: "The shard's `chunk_shape` and array's `shape` need to
    # have the same number of dimensions."
    concatenated = xr.concat(vdss, dim="time")

    assert concatenated["data"].shape == (2, 12, 12)
    assert concatenated["data"].data.metadata.shards == (1, 12, 12)
    assert concatenated["data"].data.metadata.chunks == (1, 3, 3)

    session = icechunk_repo.writable_session("main")
    concatenated.vz.to_icechunk(session.store, validate_containers=False)
    session.commit("concat sharded arrays along a new dim")

    ro_session = icechunk_repo.readonly_session("main")
    roundtrip = xr.open_zarr(ro_session.store, zarr_format=3, consolidated=False)

    assert roundtrip["data"].shape == (2, 12, 12)
    # the payoff: every inner chunk of both shards decodes to its original values
    npt.assert_array_equal(
        roundtrip["data"].values, xr.concat(datasets, dim="time")["data"].values
    )


class TestManifestGroupToIcechunk:
    @pytest.fixture
    def raw_marr(
        self, tmp_path: Path, array_v3_metadata
    ) -> Callable[..., ManifestArray]:
        """A single-chunk ManifestArray pointing at the raw bytes of ``arr``, written to a file called ``name``."""

        def _raw_marr(
            name: str,
            arr: np.ndarray,
            dimension_names: Optional[tuple[str, ...]] = None,
            attributes: Optional[dict] = None,
            fill_value: Any = 0,
        ) -> ManifestArray:
            filepath = tmp_path / name
            filepath.write_bytes(arr.tobytes())
            endian = "big" if arr.dtype.str.startswith(">") else "little"
            metadata = array_v3_metadata(
                shape=arr.shape,
                chunks=arr.shape,
                data_type=arr.dtype,
                codecs=[{"name": "bytes", "configuration": {"endian": endian}}],
                dimension_names=dimension_names,
                attributes=attributes,
                fill_value=fill_value,
            )
            manifest = ChunkManifest(
                {
                    ".".join(["0"] * arr.ndim): {
                        "path": str(filepath),
                        "offset": 0,
                        "length": arr.nbytes,
                    }
                }
            )
            return ManifestArray(metadata=metadata, chunkmanifest=manifest)

        return _raw_marr

    def test_roundtrip_nested_groups(
        self, icechunk_filestore: "IcechunkStore", raw_marr
    ):
        a = np.arange(24, dtype="<i4").reshape(4, 6)
        b = np.arange(5, dtype="<f8")
        mgroup = ManifestGroup(
            arrays={"a": raw_marr("a", a, ("y", "x"), {"units": "m"})},
            groups={
                "sub": ManifestGroup(
                    arrays={"b": raw_marr("b", b, ("t",))}, attributes={"level": 1}
                )
            },
            attributes={"title": "nested"},
        )

        ManifestStore(mgroup).to_icechunk(icechunk_filestore)

        store = icechunk_filestore
        assert zarr.open_group(store, mode="r").attrs.asdict() == {"title": "nested"}
        assert zarr.open_group(store, path="sub", mode="r").attrs.asdict() == {
            "level": 1
        }
        a_written = zarr.open_array(store, path="a", mode="r")
        assert isinstance(a_written.metadata, ArrayV3Metadata)
        assert a_written.metadata.dimension_names == ("y", "x")
        assert a_written.attrs.asdict() == {"units": "m"}
        npt.assert_array_equal(a_written[:], a)
        b_written = zarr.open_array(store, path="sub/b", mode="r")
        assert isinstance(b_written.metadata, ArrayV3Metadata)
        assert b_written.metadata.dimension_names == ("t",)
        npt.assert_array_equal(b_written[:], b)

    @pytest.mark.parametrize(
        "structure, xarray_error",
        [
            ("unnamed_array", "without dimension names"),
            # the levels of a multiscale image pyramid
            ("siblings_sharing_a_name_at_different_lengths", "conflicting sizes"),
            ("subgroup_reusing_a_parent_name", "not aligned with its parents"),
        ],
    )
    def test_writes_structure_xarray_cannot_hold(
        self,
        icechunk_filestore: "IcechunkStore",
        raw_marr,
        structure: str,
        xarray_error: str,
    ):
        long = np.arange(6, dtype="<i4")
        short = long[::2].copy()
        expected: dict[str, tuple[np.ndarray, Optional[tuple[str, ...]]]]
        if structure == "unnamed_array":
            expected = {"a": (long, None)}
            mgroup = ManifestGroup(arrays={"a": raw_marr("a", long)})
        elif structure == "siblings_sharing_a_name_at_different_lengths":
            expected = {"0": (long, ("x",)), "1": (short, ("x",))}
            mgroup = ManifestGroup(
                arrays={
                    "0": raw_marr("0", long, ("x",)),
                    "1": raw_marr("1", short, ("x",)),
                }
            )
        else:
            expected = {"a": (long, ("x",)), "sub/b": (short, ("x",))}
            mgroup = ManifestGroup(
                arrays={"a": raw_marr("a", long, ("x",))},
                groups={
                    "sub": ManifestGroup(arrays={"b": raw_marr("b", short, ("x",))})
                },
            )
        with pytest.raises(ValueError, match=xarray_error):
            mgroup.to_virtual_datatree()

        mgroup.to_icechunk(icechunk_filestore)

        for path, (values, dimension_names) in expected.items():
            written = zarr.open_array(icechunk_filestore, path=path, mode="r")
            assert isinstance(written.metadata, ArrayV3Metadata)
            assert written.metadata.dimension_names == dimension_names
            npt.assert_array_equal(written[:], values)

    @pytest.fixture
    def cf_encoded_netcdf4_file(self, tmp_path: Path) -> str:
        """Packed integers, a CF time axis and compressed chunks: where the xarray path does the most encoding."""
        filepath = tmp_path / "cf_encoded.nc"
        ds = xr.Dataset(
            {
                "air": (
                    ("time", "lat", "lon"),
                    np.random.default_rng(0).random((3, 4, 6), dtype="float32"),
                    {"units": "K", "long_name": "air temperature"},
                ),
                "packed": (("lat", "lon"), np.linspace(0, 100, 24).reshape(4, 6)),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=3),
                "lat": np.arange(4.0),
                "lon": np.arange(6.0),
            },
            attrs={"title": "CF-encoded"},
        )
        ds.to_netcdf(
            filepath,
            engine="h5netcdf",
            encoding={
                "air": {"chunksizes": (1, 4, 6), "zlib": True},
                "packed": {
                    "dtype": "int16",
                    "scale_factor": 0.01,
                    "add_offset": 0.0,
                    "_FillValue": -9999,
                },
            },
        )
        return str(filepath)

    @pytest.mark.parametrize(
        "netcdf_fixture",
        [
            "netcdf4_file_with_data_in_sibling_groups",
            "cf_encoded_netcdf4_file",
            # non-dimension coordinates, which xarray records in a group attribute this path doesn't write
            "netcdf4_file_with_2d_coords",
        ],
    )
    def test_writes_same_store_as_xarray_path(
        self,
        icechunk_repo: "Repository",
        tmp_path: Path,
        local_registry,
        netcdf_fixture: str,
        request: pytest.FixtureRequest,
    ):
        netcdf_file = request.getfixturevalue(netcdf_fixture)
        manifest_store = HDFParser()(f"file://{netcdf_file}", local_registry)

        direct_session = icechunk_repo.writable_session("main")
        manifest_store.to_icechunk(direct_session.store)
        direct_session.commit("direct")

        xarray_repo = icechunk.Repository.create(
            storage=icechunk.Storage.new_local_filesystem(str(tmp_path / "xarray")),
            config=icechunk_repo.config,
            authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
        )
        xarray_session = xarray_repo.writable_session("main")
        # nothing loaded, so both paths write only virtual refs
        manifest_store.to_virtual_datatree(loadable_variables=[]).vz.to_icechunk(
            xarray_session.store
        )
        xarray_session.commit("via xarray")

        def all_metadata(repo: "Repository") -> dict[str, dict]:
            root = zarr.open_group(
                repo.readonly_session("main").store, mode="r", zarr_format=3
            )
            members = dict(root.members(max_depth=None))
            return {
                "": root.metadata.to_dict(),
                **{path: node.metadata.to_dict() for path, node in members.items()},
            }

        direct = all_metadata(icechunk_repo)
        via_xarray = all_metadata(xarray_repo)
        assert direct.keys() == via_xarray.keys()
        for path in direct:
            if via_xarray[path]["node_type"] == "group":
                # xarray also lists each group's coordinate variables in a group attribute
                via_xarray[path]["attributes"].pop("coordinates", None)
            assert direct[path] == via_xarray[path], path

        def open_written(repo: "Repository") -> xr.DataTree:
            return xr.open_datatree(
                repo.readonly_session("main").store,  # type: ignore
                engine="zarr",
                zarr_format=3,
                consolidated=False,
            )

        with (
            open_written(icechunk_repo) as roundtrip,
            open_written(xarray_repo) as roundtrip_via_xarray,
            xr.open_datatree(netcdf_file, engine="h5netcdf") as source,
        ):
            xrt.assert_identical(roundtrip, roundtrip_via_xarray)
            # equal rather than identical: HDFParser reads some empty-string attributes back as " "
            xrt.assert_equal(roundtrip, source)

    @pytest.mark.parametrize("group", [None, "", "/a", "a", "/a/b", "a/b", "a/b/"])
    def test_write_into_group(
        self, icechunk_filestore: "IcechunkStore", raw_marr, group: Optional[str]
    ):
        a = np.arange(4, dtype="<i4")
        mgroup = ManifestGroup(
            arrays={"a": raw_marr("a", a, ("x",))},
            groups={"sub": ManifestGroup(arrays={"b": raw_marr("b", a, ("x",))})},
        )

        mgroup.to_icechunk(icechunk_filestore, group=group)

        prefix = (group or "").strip("/")
        for path in ["a", "sub/b"]:
            written = zarr.open_array(
                icechunk_filestore,
                path=f"{prefix}/{path}" if prefix else path,
                mode="r",
            )
            npt.assert_array_equal(written[:], a)

    def test_mode(self, icechunk_filestore: "IcechunkStore", raw_marr):
        a = np.arange(4, dtype="<i4")
        foo = ManifestGroup(arrays={"foo": raw_marr("foo", a, ("x",))})
        bar = ManifestGroup(arrays={"bar": raw_marr("bar", a, ("x",))})
        foo.to_icechunk(icechunk_filestore)

        with pytest.raises(ContainsGroupError):
            bar.to_icechunk(icechunk_filestore)

        bar.to_icechunk(icechunk_filestore, mode="a")
        assert set(zarr.open_group(icechunk_filestore, mode="r")) == {"foo", "bar"}

        foo.to_icechunk(icechunk_filestore, mode="w")
        assert set(zarr.open_group(icechunk_filestore, mode="r")) == {"foo"}

    @pytest.mark.parametrize(
        "existing_b, expected_error",
        [
            ("array", pytest.raises(ValueError, match="with different codecs")),
            ("group", pytest.raises(ContainsGroupError, match="later/b")),
        ],
    )
    def test_mode_a_writes_nothing_when_a_later_array_cannot_be_written(
        self,
        icechunk_repo: "Repository",
        raw_marr,
        existing_b: str,
        expected_error,
    ):
        values = np.arange(4, dtype="<i4")
        existing = ManifestGroup(arrays={"b": raw_marr("b", values, ("x",))})
        session = icechunk_repo.writable_session("main")
        if existing_b == "group":
            existing = ManifestGroup(groups={"b": existing})
        existing.to_icechunk(session.store, group="later")
        session.commit("existing group")

        # "earlier" is written before "later" would fail
        mgroup = ManifestGroup(
            groups={
                "earlier": ManifestGroup(arrays={"a": raw_marr("a", values, ("x",))}),
                "later": ManifestGroup(
                    arrays={"b": raw_marr("b2", values.astype(">i4"), ("x",))}
                ),
            },
        )
        session = icechunk_repo.writable_session("main")
        with expected_error:
            mgroup.to_icechunk(session.store, mode="a")

        assert not session.has_uncommitted_changes, session.status()

    @pytest.mark.parametrize(
        "store_kind, kwargs, error, match",
        [
            ("writable", {"group": 1}, TypeError, "group"),
            (
                "writable",
                {"last_updated_at": "2026-01-01"},
                TypeError,
                "last_updated_at",
            ),
            ("writable", {"mode": "r"}, ValueError, "mode"),
            ("memory", {}, TypeError, "expected type IcechunkStore"),
            ("read-only", {}, ValueError, "read-only"),
        ],
    )
    def test_invalid_arguments(
        self,
        icechunk_repo: "Repository",
        raw_marr,
        store_kind: str,
        kwargs: dict,
        error: type[Exception],
        match: str,
    ):
        store: "IcechunkStore | zarr.storage.MemoryStore"
        if store_kind == "memory":
            store = zarr.storage.MemoryStore()
        elif store_kind == "read-only":
            store = icechunk_repo.readonly_session("main").store
        else:
            store = icechunk_repo.writable_session("main").store
        mgroup = ManifestGroup(arrays={"a": raw_marr("a", np.arange(4), ("x",))})

        with pytest.raises(error, match=match):
            mgroup.to_icechunk(store, **kwargs)  # type: ignore[arg-type]

    def test_validate_containers(
        self, icechunk_filestore: "IcechunkStore", array_v3_metadata
    ):
        manifest = ChunkManifest(
            {"0.0": {"path": "s3://bucket/path/file.nc", "offset": 0, "length": 100}}
        )
        marr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(shape=(3, 4), chunks=(3, 4)),
        )
        # the ref without a container sits in a subgroup, and must still stop the root being written
        mgroup = ManifestGroup(groups={"sub": ManifestGroup(arrays={"foo": marr})})

        with pytest.raises(
            ValueError, match="No Virtual Chunk Container set which supports prefix"
        ):
            mgroup.to_icechunk(icechunk_filestore)

        session = icechunk_filestore.session
        assert not session.has_uncommitted_changes, session.status()
