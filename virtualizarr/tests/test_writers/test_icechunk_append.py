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
    chunks: tuple[int, int] = (3, 4),
    dtype: np.dtype = np.dtype("int32"),
    compressor: str = None,
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
        chunks=chunks,
        base_offset=base_offset,
        length=length,
    )
    zarray = ZArray(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        filters=filters,
        fill_value=fill_value,
    )
    ma = ManifestArray(chunkmanifest=manifest, zarray=zarray)
    ds = open_dataset(file_uri)
    dims = dims or list(ds.dims.keys())
    var = Variable(
        data=ma,
        dims=dims,
        encoding=encoding,
        attrs=ds[variable_name].attrs,
    )
    return Dataset(
        {variable_name: var},
    )


def test_set_single_virtual_ref_without_encoding(
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


# encoding is applied
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
            chunks=(1460, 25, 53),
            dims=["time", "lat", "lon"],
            dtype=np.dtype("int16"),
            variable_name="air",
            encoding={"_FillValue": -9999, "scale_factor": scale_factor},
            base_offset=15419,
            length=3869000,
        ),
        gen_virtual_dataset(
            file_uri=filepath2,
            shape=(1460, 25, 53),
            chunks=(1460, 25, 53),
            dims=["time", "lat", "lon"],
            dtype=np.dtype("int16"),
            variable_name="air",
            encoding={"_FillValue": -9999, "scale_factor": scale_factor},
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


# when chunking is different it fails
# There's a whole beartrap here around noticing if the last chunk is smaller than the other chunks. We should throw in that case (because zarr can't support it without variable-length chunks).
# when encoding is different it fails
# when using compression it works
# Should also test that it raises a clear error if you try to append with chunks of a different dtype etc. I would hope zarr-python would throw that for us.
