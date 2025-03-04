import time
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
import xarray as xr
import zarr
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.readers.common import separate_coords
from virtualizarr.writers.icechunk import generate_chunk_key
from virtualizarr.zarr import ZArray

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
def icechunk_repo(icechunk_storage: "Storage") -> "Repository":
    from icechunk import Repository

    repo = Repository.create(storage=icechunk_storage)
    return repo


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
        vds_with_manifest_arrays.virtualize.to_icechunk(
            icechunk_filestore, **{name: value}
        )


@pytest.mark.parametrize("group_path", [None, "", "/a", "a", "/a/b", "a/b", "a/b/"])
def test_write_new_virtual_variable(
    icechunk_filestore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    group_path: Optional[str],
):
    vds = vds_with_manifest_arrays

    vds.virtualize.to_icechunk(icechunk_filestore, group=group_path)

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
    icechunk_filestore: "IcechunkStore", simple_netcdf4: Path
):
    import xarray.testing as xrt
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
    foo = xr.Variable(data=ma, dims=["x", "y"])
    vds = xr.Dataset(
        {"foo": foo},
    )

    vds.virtualize.to_icechunk(icechunk_filestore)

    root_group = zarr.group(store=icechunk_filestore)
    array = root_group["foo"]

    # check chunk references
    # TODO we can't explicitly check that the path/offset/length is correct because
    # icechunk doesn't yet expose any get_virtual_refs method

    with (
        xr.open_zarr(store=icechunk_filestore, zarr_format=3, consolidated=False) as ds,
        xr.open_dataset(simple_netcdf4) as expected_ds,
    ):
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(array, expected_array)
        xrt.assert_identical(ds.foo, expected_ds.foo)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


def test_set_single_virtual_ref_with_encoding(
    icechunk_filestore: "IcechunkStore", netcdf4_file: Path
):
    import xarray.testing as xrt

    with xr.open_dataset(netcdf4_file) as ds:
        # We drop the coordinates because we don't have them in the zarr test case
        expected_ds = ds.drop_vars(["lon", "lat", "time"])

        # instead, for now just write out byte ranges explicitly
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
        air = xr.Variable(
            data=ma,
            dims=["time", "lat", "lon"],
            encoding={"scale_factor": 0.01},
            attrs=expected_ds["air"].attrs,
        )
        vds = xr.Dataset({"air": air}, attrs=expected_ds.attrs)

        vds.virtualize.to_icechunk(icechunk_filestore)

        root_group = zarr.group(store=icechunk_filestore)
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
            store=icechunk_filestore, zarr_format=3, consolidated=False
        ) as actual_ds:
            # Because we encode attributes, attributes may differ, for example
            # actual_range for expected_ds.air is array([185.16, 322.1 ], dtype=float32)
            # but encoded it is [185.16000366210935, 322.1000061035156]
            xrt.assert_allclose(actual_ds, expected_ds)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage
    # them now.


def test_set_grid_virtual_refs(icechunk_filestore: "IcechunkStore", netcdf4_file: Path):
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
    zarray = ZArray(
        shape=(4, 4),
        chunks=(2, 2),
        dtype=np.dtype("<i4"),
        compressor=None,
        filters=None,
        fill_value=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        zarray=zarray,
    )
    air = xr.Variable(data=ma, dims=["y", "x"])
    vds = xr.Dataset(
        {"air": air},
    )

    vds.virtualize.to_icechunk(icechunk_filestore)

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
):
    # instead for now just write out byte ranges explicitly
    manifest = ChunkManifest(
        {"0.0": {"path": str(simple_netcdf4), "offset": 6144, "length": 48}}
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

    ma_v = xr.Variable(data=ma, dims=["x", "y"])

    la_v = xr.Variable(
        dims=["x", "y"],
        data=np.random.rand(3, 4),
        attrs={"units": "km"},
    )
    vds = xr.Dataset({"air": la_v}, {"pres": ma_v})

    # Icechunk checksums currently store with second precision, so we need to make sure
    # the checksum_date is at least one second in the future
    checksum_date = datetime.now(timezone.utc) + timedelta(seconds=1)
    vds.virtualize.to_icechunk(icechunk_filestore, last_updated_at=checksum_date)

    root_group = zarr.group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, zarr.Array)
    assert air_array.shape == (3, 4)
    assert air_array.dtype == np.dtype("float64")
    assert air_array.attrs["units"] == "km"
    npt.assert_equal(air_array[:], la_v[:])

    pres_array = root_group["pres"]
    assert isinstance(pres_array, zarr.Array)
    assert pres_array.shape == (3, 4)
    assert pres_array.dtype == np.dtype("int32")

    with xr.open_dataset(simple_netcdf4) as expected_ds:
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(pres_array, expected_array)


def test_checksum(
    icechunk_filestore: "IcechunkStore",
    tmpdir: Path,
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

    ma_v = xr.Variable(data=ma, dims=["x", "y"])

    vds = xr.Dataset({"pres": ma_v})

    # Icechunk checksums currently store with second precision, so we need to make sure
    # the checksum_date is at least one second in the future
    checksum_date = datetime.now(timezone.utc) + timedelta(seconds=1)
    vds.virtualize.to_icechunk(icechunk_filestore, last_updated_at=checksum_date)

    # Fail if anything but None or a datetime is passed to last_updated_at
    with pytest.raises(TypeError):
        vds.virtualize.to_icechunk(icechunk_filestore, last_updated_at="not a datetime")  # type: ignore

    root_group = zarr.group(store=icechunk_filestore)
    pres_array = root_group["pres"]
    assert isinstance(pres_array, zarr.Array)
    assert pres_array.shape == (3, 4)
    assert pres_array.dtype == np.dtype("int32")

    with xr.open_dataset(netcdf_path) as expected_ds:
        expected_array = expected_ds["foo"].to_numpy()
        npt.assert_equal(pres_array, expected_array)

    # Now we can overwrite the simple_netcdf4 file with new data to make sure that
    # the checksum_date is being used to determine if the data is valid
    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4) * 2
    var = xr.Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    time.sleep(1)  # Make sure the checksum_date is at least one second in the future
    ds.to_netcdf(netcdf_path)

    # Now if we try to read the data back in, it should fail because the checksum_date
    # is newer than the last_updated_at
    with pytest.raises(IcechunkError):
        pres_array = root_group["pres"]
        assert isinstance(pres_array, zarr.Array)
        npt.assert_equal(pres_array, arr)


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


def generate_chunk_manifest(
    netcdf4_file: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    offset=6144,
    length=48,
) -> ChunkManifest:
    chunk_dict = {}
    num_chunks = [shape[i] // chunks[i] for i in range(len(shape))]
    offset = offset

    # Generate all possible chunk indices using Cartesian product
    for chunk_indices in product(*[range(n) for n in num_chunks]):
        chunk_index = ".".join(map(str, chunk_indices))
        chunk_dict[chunk_index] = {
            "path": netcdf4_file,
            "offset": offset,
            "length": length,
        }
        offset += length  # Increase offset for each chunk

    return ChunkManifest(chunk_dict)


def gen_virtual_variable(
    file_uri: str,
    shape: tuple[int, ...] = (3, 4),
    chunk_shape: tuple[int, ...] = (3, 4),
    dtype: np.dtype = np.dtype("int32"),
    compressor: Optional[dict] = None,
    filters: Optional[list[dict[Any, Any]]] = None,
    fill_value: Optional[str] = None,
    encoding: Optional[dict] = None,
    offset: int = 6144,
    length: int = 48,
    dims: list[str] = [],
    zarr_format: Literal[2, 3] = 2,
    attrs: dict[str, Any] = {},
) -> xr.Variable:
    manifest = generate_chunk_manifest(
        file_uri,
        shape=shape,
        chunks=chunk_shape,
        offset=offset,
        length=length,
    )
    zarray = ZArray(
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        fill_value=fill_value,
        zarr_format=zarr_format,
    )
    ma = ManifestArray(chunkmanifest=manifest, zarray=zarray)
    return xr.Variable(
        data=ma,
        dims=dims,
        encoding=encoding,
        attrs=attrs,
    )


def gen_virtual_dataset(
    file_uri: str,
    shape: tuple[int, ...] = (3, 4),
    chunk_shape: tuple[int, ...] = (3, 4),
    dtype: np.dtype = np.dtype("int32"),
    compressor: Optional[dict] = None,
    filters: Optional[list[dict[Any, Any]]] = None,
    fill_value: Optional[str] = None,
    encoding: Optional[dict] = None,
    variable_name: str = "foo",
    offset: int = 6144,
    length: int = 48,
    dims: Optional[list[str]] = None,
    zarr_format: Literal[2, 3] = 2,
    coords: Optional[xr.Coordinates] = None,
) -> xr.Dataset:
    with xr.open_dataset(file_uri) as ds:
        var = gen_virtual_variable(
            file_uri,
            shape=shape,
            chunk_shape=chunk_shape,
            dtype=dtype,
            compressor=compressor,
            filters=filters,
            fill_value=fill_value,
            encoding=encoding,
            offset=offset,
            length=length,
            dims=dims or [str(name) for name in ds.dims],
            zarr_format=zarr_format,
            attrs=ds[variable_name].attrs,
        )

        return xr.Dataset(
            {variable_name: var},
            coords=coords,
            attrs=ds.attrs,
        )


class TestAppend:
    """
    Tests for appending to existing icechunk store.
    """

    # Success cases
    ## When appending to a single virtual ref without encoding, it succeeds
    def test_append_virtual_ref_without_encoding(
        self, icechunk_repo: "Repository", simple_netcdf4: str
    ):
        import xarray.testing as xrt

        # generate virtual dataset
        vds = gen_virtual_dataset(file_uri=simple_netcdf4)
        # Commit the first virtual dataset
        writable_session = icechunk_repo.writable_session("main")
        vds.virtualize.to_icechunk(writable_session.store)
        writable_session.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        append_session = icechunk_repo.writable_session("main")

        # Append the same dataset to the same store
        vds.virtualize.to_icechunk(append_session.store, append_dim="x")
        append_session.commit("appended data")

        second_append_session = icechunk_repo.writable_session("main")
        vds.virtualize.to_icechunk(second_append_session.store, append_dim="x")
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
        self, icechunk_repo: "Repository", netcdf4_files_factory: Callable
    ):
        import xarray.testing as xrt

        scale_factor = 0.01
        encoding = {"air": {"scale_factor": scale_factor}}
        filepath1, filepath2 = netcdf4_files_factory(encoding=encoding)

        vds1, vds2 = (
            gen_virtual_dataset(
                file_uri=filepath1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                encoding={"scale_factor": scale_factor},
                offset=15419,
                length=15476000,
            ),
            gen_virtual_dataset(
                file_uri=filepath2,
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
        vds1.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines

        # Append the same dataset to the same store
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="time")
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
        self, icechunk_repo: "Repository", netcdf4_files_factory: Callable
    ):
        import xarray.testing as xrt
        from zarr.core.buffer import default_buffer_prototype

        filepath1, filepath2 = netcdf4_files_factory(
            encoding={"air": {"dtype": "float64", "chunksizes": (1460, 25, 53)}}
        )

        lon_manifest = gen_virtual_variable(
            filepath1,
            shape=(53,),
            chunk_shape=(53,),
            dtype=np.dtype("float32"),
            offset=5279,
            length=212,
            dims=["lon"],
        )
        lat_manifest = gen_virtual_variable(
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
            gen_virtual_variable(
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
            gen_virtual_dataset(
                file_uri=filepath1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=15476000,
                coords=coords1,
            ),
            gen_virtual_dataset(
                file_uri=filepath2,
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
        vds1.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        first_time_chunk_before_append = await icechunk_filestore.store.get(
            "time/c/0", prototype=default_buffer_prototype()
        )

        # Append the same dataset to the same store
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="time")
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
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_append_with_compression_succeeds(
        self,
        icechunk_repo: "Repository",
        netcdf4_files_factory: Callable,
        zarr_format: Literal[2, 3],
    ):
        import xarray.testing as xrt

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
            gen_virtual_dataset(
                file_uri=file1,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                compressor={"id": "zlib", "level": 4},
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=3936114,
                zarr_format=zarr_format,
            ),
            gen_virtual_dataset(
                file_uri=file2,
                shape=(1460, 25, 53),
                chunk_shape=(1460, 25, 53),
                compressor={"id": "zlib", "level": 4},
                dims=["time", "lat", "lon"],
                dtype=np.dtype("float64"),
                variable_name="air",
                offset=18043,
                length=3938672,
                zarr_format=zarr_format,
            ),
        )

        # Commit the compressed dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Append another dataset with compatible compression
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        vds2.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="time")
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
        self, icechunk_repo: "Repository", simple_netcdf4: str
    ):
        # Generate a virtual dataset with specific chunking
        vds = gen_virtual_dataset(file_uri=simple_netcdf4, chunk_shape=(3, 4))

        # Commit the dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Try to append dataset with different chunking, expect failure
        vds_different_chunking = gen_virtual_dataset(
            file_uri=simple_netcdf4, chunk_shape=(1, 1)
        )
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(
            ValueError, match="Cannot concatenate arrays with inconsistent chunk shapes"
        ):
            vds_different_chunking.virtualize.to_icechunk(
                icechunk_filestore_append.store, append_dim="x"
            )

    ## When encoding is different it fails
    def test_append_with_different_encoding_fails(
        self, icechunk_repo: "Repository", simple_netcdf4: str
    ):
        # Generate datasets with different encoding
        vds1 = gen_virtual_dataset(
            file_uri=simple_netcdf4, encoding={"scale_factor": 0.1}
        )
        vds2 = gen_virtual_dataset(
            file_uri=simple_netcdf4, encoding={"scale_factor": 0.01}
        )

        # Commit the first dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Try to append with different encoding, expect failure
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(
            ValueError,
            match="Cannot concatenate arrays with different values for encoding",
        ):
            vds2.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="x")

    def test_dimensions_do_not_align(
        self, icechunk_repo: "Repository", simple_netcdf4: str
    ):
        # Generate datasets with different lengths on the non-append dimension (x)
        vds1 = gen_virtual_dataset(
            # {'x': 5, 'y': 4}
            file_uri=simple_netcdf4,
            shape=(5, 4),
        )
        vds2 = gen_virtual_dataset(
            # {'x': 6, 'y': 4}
            file_uri=simple_netcdf4,
            shape=(6, 4),
        )

        # Commit the first dataset
        icechunk_filestore = icechunk_repo.writable_session("main")
        vds1.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("test commit")

        # Attempt to append dataset with different length in non-append dimension, expect failure
        icechunk_filestore_append = icechunk_repo.writable_session("main")
        with pytest.raises(ValueError, match="Cannot concatenate arrays with shapes"):
            vds2.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="y")

    def test_append_dim_not_in_dims_raises_error(
        self, icechunk_repo: "Repository", simple_netcdf4: str
    ):
        """
        Test that attempting to append with an append_dim not present in dims raises a ValueError.
        """
        vds = gen_virtual_dataset(
            file_uri=simple_netcdf4, shape=(5, 4), chunk_shape=(5, 4), dims=["x", "y"]
        )

        icechunk_filestore = icechunk_repo.writable_session("main")
        vds.virtualize.to_icechunk(icechunk_filestore.store)
        icechunk_filestore.commit("initial commit")

        # Attempt to append using a non-existent append_dim "z"
        icechunk_filestore_append = icechunk_repo.writable_session("main")

        with pytest.raises(
            ValueError,
            match="append_dim 'z' does not match any existing dataset dimensions",
        ):
            vds.virtualize.to_icechunk(icechunk_filestore_append.store, append_dim="z")


# TODO test with S3 / minio
