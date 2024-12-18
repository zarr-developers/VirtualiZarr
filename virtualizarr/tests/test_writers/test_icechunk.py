from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, cast

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
from xarray import Coordinates, Dataset, concat, open_dataset, open_zarr
from xarray.core.variable import Variable
from zarr import Array, Group, group  # type: ignore
from zarr.core.metadata import ArrayV3Metadata  # type: ignore

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.readers.common import separate_coords
from virtualizarr.writers.icechunk import dataset_to_icechunk, generate_chunk_key
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    from icechunk import IcechunkStore, StorageConfig  # type: ignore[import-not-found]


@pytest.fixture(scope="function")
def icechunk_filestore(tmpdir) -> "IcechunkStore":
    from icechunk import IcechunkStore, StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO if icechunk exposed a synchronous version of .open then we wouldn't need to use asyncio.run here
    # TODO is this the correct mode to use?
    store = IcechunkStore.create(storage=storage, mode="w")

    # TODO instead yield store then store.close() ??
    return store


def test_write_new_virtual_variable(
    icechunk_filestore: "IcechunkStore", vds_with_manifest_arrays: Dataset
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

    ds = open_zarr(store=icechunk_filestore, zarr_format=3, consolidated=False)
    # TODO: Check using xarray.testing.assert_identical
    xrt.assert_identical(ds.foo, expected_ds.foo)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


def test_set_single_virtual_ref_with_encoding(
    icechunk_filestore: "IcechunkStore", netcdf4_file: Path
):
    import xarray.testing as xrt

    expected_ds = open_dataset(netcdf4_file).drop_vars(["lon", "lat", "time"])
    # these attributes encode floats different and I am not sure why, but its not important enough to block everything
    expected_ds.air.attrs.pop("actual_range")

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
    air = Variable(
        data=ma,
        dims=["time", "lat", "lon"],
        encoding={"scale_factor": 0.01},
        attrs=expected_ds.air.attrs,
    )
    vds = Dataset({"air": air}, attrs=expected_ds.attrs)

    dataset_to_icechunk(vds, icechunk_filestore)

    root_group = group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, Array)

    # check array metadata
    assert air_array.shape == (2920, 25, 53)
    assert air_array.chunks == (2920, 25, 53)
    assert air_array.dtype == np.dtype("int16")
    assert air_array.attrs["scale_factor"] == 0.01

    # check chunk references
    # TODO we can't explicitly check that the path/offset/length is correct because icechunk doesn't yet expose any get_virtual_refs method

    # Load in the dataset, we drop the coordinates because we don't have them in the zarr test case
    # Check with xarray
    ds = open_zarr(store=icechunk_filestore, zarr_format=3, consolidated=False)
    xrt.assert_identical(ds, expected_ds)

    # note: we don't need to test that committing works, because now we have confirmed
    # the refs are in the store (even uncommitted) it's icechunk's problem to manage them now.


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
    air = Variable(data=ma, dims=["y", "x"])
    vds = Dataset(
        {"air": air},
    )

    dataset_to_icechunk(vds, icechunk_filestore)

    root_group = group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, Array)

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
    icechunk_filestore: "IcechunkStore", simple_netcdf4: Path
):
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

    ma_v = Variable(data=ma, dims=["x", "y"])

    la_v = Variable(
        dims=["x", "y"],
        data=np.random.rand(3, 4),
        attrs={"units": "km"},
    )
    vds = Dataset({"air": la_v}, {"pres": ma_v})

    dataset_to_icechunk(vds, icechunk_filestore)

    root_group = group(store=icechunk_filestore)
    air_array = root_group["air"]
    assert isinstance(air_array, Array)
    assert air_array.shape == (3, 4)
    assert air_array.dtype == np.dtype("float64")
    assert air_array.attrs["units"] == "km"
    npt.assert_equal(air_array[:], la_v[:])

    pres_array = root_group["pres"]
    assert isinstance(pres_array, Array)
    assert pres_array.shape == (3, 4)
    assert pres_array.dtype == np.dtype("int32")
    expected_ds = open_dataset(simple_netcdf4)
    expected_array = expected_ds["foo"].to_numpy()
    npt.assert_equal(pres_array, expected_array)


def test_generate_chunk_key_no_offset():
    # Test case without any offset (append_axis and existing_num_chunks are None)
    index = (1, 2, 3)
    result = generate_chunk_key(index)
    assert result == "1/2/3", "The chunk key should match the index without any offset."


def test_generate_chunk_key_with_offset():
    # Test case with offset on append_axis 1
    index = (1, 2, 3)
    append_axis = 1
    existing_num_chunks = 5
    result = generate_chunk_key(
        index, append_axis=append_axis, existing_num_chunks=existing_num_chunks
    )
    assert result == "1/7/3", "The chunk key should offset the second index by 5."


def test_generate_chunk_key_zero_offset():
    # Test case where existing_num_chunks is 0 (no offset should be applied)
    index = (4, 5, 6)
    append_axis = 1
    existing_num_chunks = 0
    result = generate_chunk_key(
        index, append_axis=append_axis, existing_num_chunks=existing_num_chunks
    )
    assert (
        result == "4/5/6"
    ), "No offset should be applied when existing_num_chunks is 0."


def test_generate_chunk_key_append_axis_out_of_bounds():
    # Edge case where append_axis is out of bounds
    index = (3, 4)
    append_axis = 2  # This is out of bounds for a 2D index
    with pytest.raises(ValueError):
        generate_chunk_key(index, append_axis=append_axis, existing_num_chunks=1)


@pytest.fixture(scope="function")
def icechunk_storage(tmpdir) -> "StorageConfig":
    from icechunk import StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO instead yield store then store.close() ??
    return storage


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
) -> Variable:
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
    return Variable(
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
    coords: Optional[Coordinates] = None,
) -> Dataset:
    ds = open_dataset(file_uri)
    ds_dims: list[str] = cast(list[str], list(ds.dims))
    dims = dims or ds_dims
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
        dims=dims,
        zarr_format=zarr_format,
        attrs=ds[variable_name].attrs,
    )
    return Dataset(
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
        self, icechunk_storage: "StorageConfig", simple_netcdf4: str
    ):
        import xarray.testing as xrt
        from icechunk import IcechunkStore

        # generate virtual dataset
        vds = gen_virtual_dataset(file_uri=simple_netcdf4)
        # create the icechunk store and commit the first virtual dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds, icechunk_filestore)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines

        # Append the same dataset to the same store
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        dataset_to_icechunk(vds, icechunk_filestore_append, append_dim="x")
        icechunk_filestore_append.commit("appended data")
        dataset_to_icechunk(vds, icechunk_filestore_append, append_dim="x")
        icechunk_filestore_append.commit("appended data again")
        array = open_zarr(icechunk_filestore_append, consolidated=False, zarr_format=3)

        expected_ds = open_dataset(simple_netcdf4)
        expected_array = concat([expected_ds, expected_ds, expected_ds], dim="x")
        xrt.assert_identical(array, expected_array)

    def test_append_virtual_ref_with_encoding(
        self, icechunk_storage: "StorageConfig", netcdf4_files_factory: Callable
    ):
        import xarray.testing as xrt
        from icechunk import IcechunkStore

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

        # create the icechunk store and commit the first virtual dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds1, icechunk_filestore)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines

        # Append the same dataset to the same store
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="time")
        icechunk_filestore_append.commit("appended data")
        new_ds = open_zarr(icechunk_filestore_append, consolidated=False, zarr_format=3)

        expected_ds1, expected_ds2 = open_dataset(filepath1), open_dataset(filepath2)
        expected_ds = concat([expected_ds1, expected_ds2], dim="time").drop_vars(
            ["time", "lat", "lon"], errors="ignore"
        )
        # Because we encode attributes, attributes may differ, for example
        # actual_range for expected_ds.air is array([185.16, 322.1 ], dtype=float32)
        # but encoded it is [185.16000366210935, 322.1000061035156]
        xrt.assert_equal(new_ds, expected_ds)

    ## When appending to a virtual ref with encoding, it succeeds
    @pytest.mark.asyncio
    async def test_append_with_multiple_root_arrays(
        self, icechunk_storage: "StorageConfig", netcdf4_files_factory: Callable
    ):
        import xarray.testing as xrt
        from icechunk import IcechunkStore

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

        # create the icechunk store and commit the first virtual dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds1, icechunk_filestore)
        icechunk_filestore.commit(
            "test commit"
        )  # need to commit it in order to append to it in the next lines
        new_ds = open_zarr(icechunk_filestore, consolidated=False, zarr_format=3)
        first_time_chunk_before_append = await icechunk_filestore._store.get("time/c/0")

        # Append the same dataset to the same store
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="time")
        icechunk_filestore_append.commit("appended data")
        assert (
            await icechunk_filestore_append._store.get("time/c/0")
        ) == first_time_chunk_before_append
        new_ds = open_zarr(icechunk_filestore_append, consolidated=False, zarr_format=3)

        expected_ds1, expected_ds2 = open_dataset(filepath1), open_dataset(filepath2)
        expected_ds = concat([expected_ds1, expected_ds2], dim="time")
        xrt.assert_equal(new_ds, expected_ds)

    # When appending to a virtual ref with compression, it succeeds
    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_append_with_compression_succeeds(
        self,
        icechunk_storage: "StorageConfig",
        netcdf4_files_factory: Callable,
        zarr_format: Literal[2, 3],
    ):
        import xarray.testing as xrt
        from icechunk import IcechunkStore

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

        # Create icechunk store and commit the compressed dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds1, icechunk_filestore)
        icechunk_filestore.commit("test commit")

        # Append another dataset with compatible compression
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="time")
        icechunk_filestore_append.commit("appended data")
        updated_ds = open_zarr(
            store=icechunk_filestore_append, consolidated=False, zarr_format=3
        )

        expected_ds1, expected_ds2 = open_dataset(file1), open_dataset(file2)
        expected_ds = concat([expected_ds1, expected_ds2], dim="time")
        expected_ds = expected_ds.drop_vars(["lon", "lat", "time"], errors="ignore")
        xrt.assert_equal(updated_ds, expected_ds)

    ## When chunk shapes are different it fails
    def test_append_with_different_chunking_fails(
        self, icechunk_storage: "StorageConfig", simple_netcdf4: str
    ):
        from icechunk import IcechunkStore

        # Generate a virtual dataset with specific chunking
        vds = gen_virtual_dataset(file_uri=simple_netcdf4, chunk_shape=(3, 4))

        # Create icechunk store and commit the dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds, icechunk_filestore)
        icechunk_filestore.commit("test commit")

        # Try to append dataset with different chunking, expect failure
        vds_different_chunking = gen_virtual_dataset(
            file_uri=simple_netcdf4, chunk_shape=(1, 1)
        )
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        with pytest.raises(
            ValueError, match="Cannot concatenate arrays with inconsistent chunk shapes"
        ):
            dataset_to_icechunk(
                vds_different_chunking, icechunk_filestore_append, append_dim="x"
            )

    ## When encoding is different it fails
    def test_append_with_different_encoding_fails(
        self, icechunk_storage: "StorageConfig", simple_netcdf4: str
    ):
        from icechunk import IcechunkStore

        # Generate datasets with different encoding
        vds1 = gen_virtual_dataset(
            file_uri=simple_netcdf4, encoding={"scale_factor": 0.1}
        )
        vds2 = gen_virtual_dataset(
            file_uri=simple_netcdf4, encoding={"scale_factor": 0.01}
        )

        # Create icechunk store and commit the first dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds1, icechunk_filestore)
        icechunk_filestore.commit("test commit")

        # Try to append with different encoding, expect failure
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        with pytest.raises(
            ValueError,
            match="Cannot concatenate arrays with different values for encoding",
        ):
            dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="x")

    def test_dimensions_do_not_align(
        self, icechunk_storage: "StorageConfig", simple_netcdf4: str
    ):
        from icechunk import IcechunkStore

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

        # Create icechunk store and commit the first dataset
        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds1, icechunk_filestore)
        icechunk_filestore.commit("test commit")

        # Attempt to append dataset with different length in non-append dimension, expect failure
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        with pytest.raises(ValueError, match="Cannot concatenate arrays with shapes"):
            dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="y")

    def test_append_dim_not_in_dims_raises_error(
        self, icechunk_storage: "StorageConfig", simple_netcdf4: str
    ):
        """
        Test that attempting to append with an append_dim not present in dims raises a ValueError.
        """
        from icechunk import IcechunkStore

        vds = gen_virtual_dataset(
            file_uri=simple_netcdf4, shape=(5, 4), chunk_shape=(5, 4), dims=["x", "y"]
        )

        icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
        dataset_to_icechunk(vds, icechunk_filestore)
        icechunk_filestore.commit("initial commit")

        # Attempt to append using a non-existent append_dim "z"
        icechunk_filestore_append = IcechunkStore.open_existing(
            storage=icechunk_storage, read_only=False
        )
        with pytest.raises(
            ValueError,
            match="append_dim z does not match any existing dataset dimensions",
        ):
            dataset_to_icechunk(vds, icechunk_filestore_append, append_dim="z")


# TODO test writing to a group that isn't the root group

# TODO test with S3 / minio
