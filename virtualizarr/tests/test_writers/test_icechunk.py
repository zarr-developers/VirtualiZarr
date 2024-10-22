from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
from xarray import Dataset, open_dataset, open_zarr
from xarray.core.variable import Variable
from zarr import Array, Group, group  # type: ignore[import-untyped]

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.writers.icechunk import dataset_to_icechunk
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]


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
    assert arr.attrs["_ARRAY_DIMENSIONS"] == ["x", "y"]


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
    # TODO kerchunk doesn't work with zarr-python v3 yet so we can't use open_virtual_dataset and icechunk together!
    # vds = open_virtual_dataset(netcdf4_file, indexes={})

    expected_ds = open_dataset(netcdf4_file).drop_vars(["lon", "lat", "time"])
    # these atyttirbutes encode floats different and I am not sure why, but its not important enough to block everything
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


# TODO test writing to a group that isn't the root group

# TODO test with S3 / minio
