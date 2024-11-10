from typing import TYPE_CHECKING

import pytest

pytest.importorskip("icechunk")

import numpy as np
import numpy.testing as npt
from xarray import Dataset, open_dataset
from xarray.core.variable import Variable
from zarr import group  # type: ignore[import-untyped]

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.writers.icechunk import dataset_to_icechunk
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    from icechunk import StorageConfig  # type: ignore[import-not-found]


@pytest.fixture(scope="function")
def icechunk_storage(tmpdir) -> "StorageConfig":
    from icechunk import StorageConfig

    storage = StorageConfig.filesystem(str(tmpdir))

    # TODO instead yield store then store.close() ??
    return storage


def generate_chunk_manifest(
    netcdf4_file: str,
    shape: tuple[int, int] = (3, 4),
    chunks: tuple[int, int] = (3, 4),
    base_offset=6144,
    length=48,
) -> ChunkManifest:
    chunk_dict = {}
    num_chunks_x = shape[0] // chunks[0]
    num_chunks_y = shape[1] // chunks[1]
    if len(shape) == 3:
        num_chunks_z = shape[2] // chunks[2]
    offset = base_offset

    for i in range(num_chunks_x):
        for j in range(num_chunks_y):
            if len(shape) == 3:
                for k in range(num_chunks_z):
                    chunk_index = f"{i}.{j}.{k}"
                    chunk_dict[chunk_index] = {
                        "path": netcdf4_file,
                        "offset": offset,
                        "length": length,
                    }
                    offset += length
            else:
                chunk_index = f"{i}.{j}"
                chunk_dict[chunk_index] = {
                    "path": netcdf4_file,
                    "offset": offset,
                    "length": length,
                }
                offset += length  # Increase offset for each chunk
    return ChunkManifest(chunk_dict)


def gen_virtual_dataset(
    file_uri: str,
    shape: tuple[int, int] = (3, 4),
    chunk_shape: tuple[int, int] = (3, 4),
    dtype: np.dtype = np.dtype("int32"),
    compressor: dict = None,
    filters: str = None,
    fill_value: str = None,
    encoding: dict = None,
    variable_name: str = "foo",
    base_offset: int = 6144,
    length: int = 48,
    dims: list[str] = None,
):
    manifest = generate_chunk_manifest(
        file_uri,
        shape=shape,
        chunks=chunk_shape,
        base_offset=base_offset,
        length=length,
    )
    zarray = ZArray(
        shape=shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        fill_value=fill_value,
    )
    ma = ManifestArray(chunkmanifest=manifest, zarray=zarray)
    ds = open_dataset(file_uri)
    dims = dims or ds.sizes.keys()
    var = Variable(
        data=ma,
        dims=dims,
        encoding=encoding,
        attrs=ds[variable_name].attrs,
    )
    return Dataset(
        {variable_name: var},
    )


# Success cases


## When appending to a single virtual ref without encoding, it succeeds
def test_append_virtual_ref_without_encoding(
    icechunk_storage: "StorageConfig", simple_netcdf4: str
):
    import xarray as xr
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
        storage=icechunk_storage, mode="a"
    )
    dataset_to_icechunk(vds, icechunk_filestore_append, append_dim="x")

    root_group = group(store=icechunk_filestore_append)
    array = root_group["foo"]

    expected_ds = open_dataset(simple_netcdf4)
    expected_array = xr.concat(
        [expected_ds["foo"], expected_ds["foo"]], dim="x"
    ).to_numpy()
    npt.assert_equal(array, expected_array)


## When appending to a virtual ref with encoding, it succeeds
def test_append_virtual_ref_with_encoding(
    icechunk_storage: "StorageConfig", netcdf4_files: tuple[str, str]
):
    import xarray as xr
    from icechunk import IcechunkStore

    # generate virtual dataset
    filepath1, filepath2 = netcdf4_files
    scale_factor = 0.01
    vds1, vds2 = (
        gen_virtual_dataset(
            file_uri=filepath1,
            shape=(1460, 25, 53),
            chunk_shape=(1460, 25, 53),
            dims=["time", "lat", "lon"],
            dtype=np.dtype("float64"),
            variable_name="air",
            encoding={"scale_factor": scale_factor},
            base_offset=15419,
            length=3869000,
        ),
        gen_virtual_dataset(
            file_uri=filepath2,
            shape=(1460, 25, 53),
            chunk_shape=(1460, 25, 53),
            dims=["time", "lat", "lon"],
            dtype=np.dtype("float64"),
            variable_name="air",
            encoding={"scale_factor": scale_factor},
            base_offset=15419,
            length=3869000,
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
        storage=icechunk_storage, mode="a"
    )
    dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="time")

    root_group = group(store=icechunk_filestore_append)
    array = root_group["air"]
    expected_ds1, expected_ds2 = open_dataset(filepath1), open_dataset(filepath2)
    expected_array = xr.concat(
        [expected_ds1["air"], expected_ds2["air"]], dim="time"
    ).to_numpy()
    npt.assert_equal(array.get_basic_selection() * scale_factor, expected_array)


## When appending to a virtual ref with compression, it succeeds
def test_append_with_compression_succeeds(
    icechunk_storage: "StorageConfig", compressed_netcdf4_files: str
):
    import xarray as xr
    from icechunk import IcechunkStore

    file1, file2 = compressed_netcdf4_files
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
            base_offset=23214,
            length=3936114,
        ),
        gen_virtual_dataset(
            file_uri=file2,
            shape=(1460, 25, 53),
            chunk_shape=(1460, 25, 53),
            compressor={"id": "zlib", "level": 4},
            dims=["time", "lat", "lon"],
            dtype=np.dtype("float64"),
            variable_name="air",
            base_offset=23214,
            length=3938672,
        ),
    )

    # Create icechunk store and commit the compressed dataset
    icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
    dataset_to_icechunk(vds1, icechunk_filestore)
    icechunk_filestore.commit("test commit")

    # Append another dataset with compatible compression
    icechunk_filestore_append = IcechunkStore.open_existing(
        storage=icechunk_storage, mode="a"
    )
    dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="time")
    root_group = group(store=icechunk_filestore_append)
    array = root_group["air"]

    expected_ds1, expected_ds2 = open_dataset(file1), open_dataset(file2)
    expected_array = xr.concat(
        [expected_ds1["air"], expected_ds2["air"]], dim="time"
    ).to_numpy()
    npt.assert_equal(array, expected_array)


## When chunk shapes are different it fails
def test_append_with_different_chunking_fails(
    icechunk_storage: "StorageConfig", simple_netcdf4: str
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
        storage=icechunk_storage, mode="a"
    )
    with pytest.raises(
        ValueError, match="Cannot concatenate arrays with inconsistent chunk shapes"
    ):
        dataset_to_icechunk(
            vds_different_chunking, icechunk_filestore_append, append_dim="x"
        )


## When encoding is different it fails
# @pytest.mark.skip(reason="working on this")
def test_append_with_different_encoding_fails(
    icechunk_storage: "StorageConfig", simple_netcdf4: str
):
    from icechunk import IcechunkStore

    # Generate datasets with different encoding
    vds1 = gen_virtual_dataset(file_uri=simple_netcdf4, encoding={"scale_factor": 0.1})
    vds2 = gen_virtual_dataset(file_uri=simple_netcdf4, encoding={"scale_factor": 0.01})

    # Create icechunk store and commit the first dataset
    icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
    dataset_to_icechunk(vds1, icechunk_filestore)
    icechunk_filestore.commit("test commit")

    # Try to append with different encoding, expect failure
    icechunk_filestore_append = IcechunkStore.open_existing(
        storage=icechunk_storage, mode="a"
    )
    with pytest.raises(
        ValueError, match="Cannot concatenate arrays with different values for encoding"
    ):
        dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="x")


# When sizes of other dimensions are different, it fails
@pytest.mark.skip(reason="working on this")
def test_other_dimensions_different_length_fails(
    icechunk_storage: "StorageConfig", simple_netcdf4: str
):
    from icechunk import IcechunkStore

    # Generate datasets with different lengths in non-append dimensions
    vds1 = gen_virtual_dataset(file_uri=simple_netcdf4, shape=(5, 4))  # shape (5, 4)
    vds2 = gen_virtual_dataset(file_uri=simple_netcdf4, shape=(6, 4))  # shape (6, 4)

    # Create icechunk store and commit the first dataset
    icechunk_filestore = IcechunkStore.create(storage=icechunk_storage)
    dataset_to_icechunk(vds1, icechunk_filestore)
    icechunk_filestore.commit("test commit")

    # Attempt to append dataset with different length in non-append dimension, expect failure
    icechunk_filestore_append = IcechunkStore.open_existing(
        storage=icechunk_storage, mode="a"
    )
    with pytest.raises(
        ValueError, match="incompatible lengths in non-append dimensions"
    ):
        dataset_to_icechunk(vds2, icechunk_filestore_append, append_dim="x")
