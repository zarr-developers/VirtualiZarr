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
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX
from virtualizarr.writers.icechunk import generate_chunk_key
from virtualizarr.xarray import separate_coords

icechunk = pytest.importorskip("icechunk")

if TYPE_CHECKING:
    from icechunk import (  # type: ignore[import-not-found]
        IcechunkStore,
        Repository,
        Storage,
        Diff,
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

    vds.vz.to_icechunk(icechunk_filestore, group=group_path)

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
    # TODO somehow this is broken by setting the dimension names???
    # assert dict(arr.attrs) == {"units": "km"}

    # check dimensions
    if isinstance(arr.metadata, ArrayV3Metadata):
        assert arr.metadata.dimension_names == ("x", "y")


def test_set_single_virtual_ref_without_encoding(
    icechunk_filestore: "IcechunkStore",
    icechunk_repo: "Repository",
    simple_netcdf4: Path,
    array_v3_metadata,
):
    # TODO kerchunk doesn't work with zarr-python v3 yet so we can't use open_virtual_dataset and icechunk together!
    # vds = open_virtual_dataset(netcdf4_file, indexes={})

    # instead for now just write out byte ranges explicitly
    manifest = ChunkManifest(
        {"0.0": {"path": simple_netcdf4, "offset": 6144, "length": 48}}
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
    foo = xr.Variable(data=ma, dims=["x", "y"])
    vds = xr.Dataset(
        {"foo": foo},
    )

    vds.vz.to_icechunk(icechunk_filestore)

    icechunk_filestore.session.commit("test")

    icechunk_readonly_session = icechunk_repo.readonly_session("main")
    root_group = zarr.open_group(store=icechunk_readonly_session.store, mode="r")
    array = root_group["foo"]

    # check chunk references
    # TODO we can't explicitly check that the path/offset/length is correct because
    # icechunk doesn't yet expose any get_virtual_refs method

    with (
        xr.open_zarr(
            store=icechunk_readonly_session.store, zarr_format=3, consolidated=False
        ) as ds,
        xr.open_dataset(simple_netcdf4) as expected_ds,
    ):
        expected_array = expected_ds["foo"].to_numpy()

        npt.assert_equal(array, expected_array)
        xrt.assert_identical(ds.foo, expected_ds.foo)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


def test_set_single_virtual_ref_with_encoding(
    icechunk_filestore: "IcechunkStore",
    icechunk_repo: "Repository",
    netcdf4_file: Path,
    array_v3_metadata,
):
    with xr.open_dataset(netcdf4_file) as ds:
        # We drop the coordinates because we don't have them in the zarr test case
        expected_ds = ds.drop_vars(["lon", "lat", "time"])

        # instead, for now just write out byte ranges explicitly
        manifest = ChunkManifest(
            {"0.0.0": {"path": netcdf4_file, "offset": 15419, "length": 7738000}}
        )
        metadata = array_v3_metadata(
            shape=(2920, 25, 53),
            chunks=(2920, 25, 53),
            codecs=None,
            data_type=np.dtype("int16"),
        )
        ma = ManifestArray(
            chunkmanifest=manifest,
            metadata=metadata,
        )
        air = xr.Variable(
            data=ma,
            dims=["time", "lat", "lon"],
            encoding={"scale_factor": 0.01},
            attrs=expected_ds["air"].attrs,
        )
        vds = xr.Dataset({"air": air}, attrs=expected_ds.attrs)

        vds.vz.to_icechunk(icechunk_filestore)

        icechunk_filestore.session.commit("test")

        icechunk_readonly_session = icechunk_repo.readonly_session("main")
        root_group = zarr.open_group(store=icechunk_readonly_session.store, mode="r")
        air_array = root_group["air"]
        assert isinstance(air_array, zarr.Array)

        # check array metadata
        assert air_array.shape == (2920, 25, 53)
        assert air_array.chunks == (2920, 25, 53)
        assert air_array.dtype == np.dtype("int16")
        assert air_array.attrs["scale_factor"] == 0.01

        # check chunk references
        # TODO we can't explicitly check that the path/offset/length is correct because
        # icechunk doesn't yet expose any get_virtual_refs method

        # check the data
        with xr.open_zarr(
            store=icechunk_readonly_session.store, zarr_format=3, consolidated=False
        ) as actual_ds:
            # Because we encode attributes, attributes may differ, for example
            # actual_range for expected_ds.air is array([185.16, 322.1 ], dtype=float32)
            # but encoded it is [185.16000366210935, 322.1000061035156]
            xrt.assert_allclose(actual_ds, expected_ds)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage
    # them now.


def test_set_grid_virtual_refs(
    icechunk_filestore: "IcechunkStore", netcdf4_file: Path, array_v3_metadata
):
    # TODO kerchunk doesn't work with zarr-python v3 yet so we can't use open_virtual_dataset and icechunk together!
    # vds = open_virtual_dataset(netcdf4_file, indexes={})

    with open(netcdf4_file, "rb") as f:
        f.seek(200)
        actual_data = f.read(64)

    # instead for now just write out random byte ranges explicitly
    manifest = ChunkManifest(
        {
            "0.0": {"path": netcdf4_file, "offset": 200, "length": 16},
            "0.1": {"path": netcdf4_file, "offset": 216, "length": 16},
            "1.0": {"path": netcdf4_file, "offset": 232, "length": 16},
            "1.1": {"path": netcdf4_file, "offset": 248, "length": 16},
        }
    )
    metadata = array_v3_metadata(
        shape=(4, 4),
        chunks=(2, 2),
        codecs=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    air = xr.Variable(data=ma, dims=["y", "x"])
    vds = xr.Dataset(
        {"air": air},
    )

    vds.vz.to_icechunk(icechunk_filestore)

    root_group = zarr.group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, zarr.Array)

    # check array metadata
    assert air_array.shape == (4, 4)
    assert air_array.chunks == (2, 2)
    assert air_array.dtype == np.dtype("int32")

    # check chunk references
    npt.assert_equal(
        air_array[:2, :2], np.frombuffer(actual_data[:16], "<i4").reshape(2, 2)
    )
    npt.assert_equal(
        air_array[:2, 2:], np.frombuffer(actual_data[16:32], "<i4").reshape(2, 2)
    )
    npt.assert_equal(
        air_array[2:, :2], np.frombuffer(actual_data[32:48], "<i4").reshape(2, 2)
    )
    npt.assert_equal(
        air_array[2:, 2:], np.frombuffer(actual_data[48:], "<i4").reshape(2, 2)
    )


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
    tmpdir: Path,
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
    diff = session.status()
    assert diff_is_empty(diff), diff


def diff_is_empty(diff: "Diff") -> bool:
    # TODO would be nicer to implement __bool__ on icechunk's Diff class
    return not any(
        [
            bool(diff.deleted_arrays),
            bool(diff.deleted_groups),
            bool(diff.new_arrays),
            bool(diff.new_groups),
            bool(diff.updated_arrays),
            bool(diff.updated_chunks),
            bool(diff.updated_groups),
        ]
    )


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
    vds.vz.to_icechunk(icechunk_filestore)
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
        self,
        icechunk_repo: "Repository",
        simple_netcdf4: str,
        virtual_dataset: Callable,
    ):
        # generate virtual dataset
        vds = virtual_dataset(url=simple_netcdf4)
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
            xr.open_zarr(
                read_session.store, consolidated=False, zarr_format=3
            ) as array,
            xr.open_dataset(simple_netcdf4) as expected_ds,
        ):
            expected_array = xr.concat([expected_ds, expected_ds, expected_ds], dim="x")
            xrt.assert_identical(array, expected_array)

    def test_append_virtual_ref_with_encoding(
        self,
        icechunk_repo: "Repository",
        netcdf4_files_factory: Callable,
        virtual_dataset: Callable,
    ):
        scale_factor = 0.01
        encoding = {"air": {"scale_factor": scale_factor}}
        filepath1, filepath2 = netcdf4_files_factory(encoding=encoding)
        vds1, vds2 = (
            virtual_dataset(
                url=filepath1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                encoding={"scale_factor": scale_factor},
                offset=15419,
                length=15476000,
            ),
            virtual_dataset(
                url=filepath2,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                encoding={"scale_factor": scale_factor},
                offset=15419,
                length=15476000,
            ),
        )

        # Commit the first virtual dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines

        # Append the same dataset to the same store
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.vz.to_icechunk(icechunk_filestore_append.store, append_dim="time")
        icechunk_filestore_append.commit("appended data")

        with (
            xr.open_dataset(filepath1) as expected_ds1,
            xr.open_dataset(filepath2) as expected_ds2,
            xr.open_zarr(
                icechunk_filestore_append.store, consolidated=False, zarr_format=3
            ) as new_ds,
        ):
            expected_ds = xr.concat([expected_ds1, expected_ds2], dim="time").drop_vars(
                ["time", "lat", "lon"], errors="ignore"
            )
            xrt.assert_equal(new_ds, expected_ds)

    ## When appending to a virtual ref with encoding, it succeeds
    @pytest.mark.asyncio
    async def test_append_with_multiple_root_arrays(
        self,
        icechunk_repo: "Repository",
        netcdf4_files_factory: Callable,
        virtual_variable: Callable,
        virtual_dataset: Callable,
    ):
        filepath1, filepath2 = netcdf4_files_factory(
            encoding={"air": {"dtype": "float64", "chunksizes": (1460, 25, 53)}}
        )

        lon_manifest = virtual_variable(
            filepath1,
            shape=(53,),
            chunk_shape=(53,),
            dtype=np.dtype("float32"),
            offset=5279,
            length=212,
            dims=["lon"],
        )
        lat_manifest = virtual_variable(
            filepath1,
            shape=(25,),
            chunk_shape=(25,),
            dtype=np.dtype("float32"),
            offset=5179,
            length=100,
            dims=["lat"],
        )
        time_attrs = {
            "standard_name": "time",
            "long_name": "Time",
            "units": "hours since 1800-01-01",
            "calendar": "standard",
        }
        time_manifest1, time_manifest2 = [
            virtual_variable(
                filepath,
                shape=(1460,),
                chunk_shape=(1460,),
                dtype=np.dtype("float32"),
                offset=15498221,
                length=5840,
                dims=["time"],
                attrs=time_attrs,
            )
            for filepath in [filepath1, filepath2]
        ]
        [[_, coords1], [_, coords2]] = [
            separate_coords(
                vars={"time": time_manifest, "lat": lat_manifest, "lon": lon_manifest},
                indexes={},
                coord_names=[],
            )
            for time_manifest in [time_manifest1, time_manifest2]
        ]
        vds1, vds2 = (
            virtual_dataset(
                url=filepath1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=15476000,
                coords=coords1,
            ),
            virtual_dataset(
                url=filepath2,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=15476000,
                coords=coords2,
            ),
        )

        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        first_time_chunk_before_append = await icechunk_filestore.store.get(
            "time/c/0", prototype=default_buffer_prototype()
        )

        # Append the same dataset to the same store
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.vz.to_icechunk(icechunk_filestore_append.store, append_dim="time")
        icechunk_filestore_append.commit("appended data")
        assert (
            await icechunk_filestore_append.store.get(
                "time/c/0", prototype=default_buffer_prototype()
            )
        ) == first_time_chunk_before_append

        with (
            xr.open_zarr(
                icechunk_filestore_append.store, consolidated=False, zarr_format=3
            ) as ds,
            xr.open_dataset(filepath1) as expected_ds1,
            xr.open_dataset(filepath2) as expected_ds2,
        ):
            expected_ds = xr.concat([expected_ds1, expected_ds2], dim="time")
            xrt.assert_equal(ds, expected_ds)

    # When appending to a virtual ref with compression, it succeeds
    def test_append_with_compression_succeeds(
        self,
        icechunk_repo: "Repository",
        netcdf4_files_factory: Callable,
        virtual_dataset: Callable,
    ):
        encoding = {
            "air": {
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1460, 25, 53),
                "shuffle": False,
            }
        }
        file1, file2 = netcdf4_files_factory(encoding=encoding)
        # Generate compressed dataset
        vds1, vds2 = (
            virtual_dataset(
                url=file1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                codecs=[
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "numcodecs.zlib", "configuration": {"level": 4}},
                ],
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=3936114,
            ),
            virtual_dataset(
                url=file2,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                codecs=[
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "numcodecs.zlib", "configuration": {"level": 4}},
                ],
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=3938672,
            ),
        )

        # Commit the compressed dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.vz.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Append another dataset with compatible compression
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.vz.to_icechunk(icechunk_filestore_append.store, append_dim="time")
        icechunk_filestore_append.commit("appended data")
        with (
            xr.open_zarr(
                store=icechunk_filestore_append.store, consolidated=False, zarr_format=3
            ) as ds,
            xr.open_dataset(file1) as expected_ds1,
            xr.open_dataset(file2) as expected_ds2,
        ):
            expected_ds = xr.concat([expected_ds1, expected_ds2], dim="time")
            expected_ds = expected_ds.drop_vars(["lon", "lat", "time"], errors="ignore")
            xrt.assert_equal(ds, expected_ds)

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
