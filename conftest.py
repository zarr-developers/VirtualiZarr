import itertools
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import h5py
import numpy as np
import pytest
import xarray as xr
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.utils import ceildiv


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def pytest_runtest_setup(item):
    # based on https://stackoverflow.com/questions/47559524
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )


@pytest.fixture
def empty_netcdf4_file(tmp_path: Path) -> str:
    filepath = tmp_path / "empty.nc"

    # Set up example xarray dataset
    with xr.Dataset() as ds:  # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_file(tmp_path: Path) -> str:
    filepath = tmp_path / "air.nc"

    # Set up example xarray dataset
    with xr.tutorial.open_dataset("air_temperature") as ds:
        # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_file_with_data_in_multiple_groups(tmp_path: Path) -> str:
    filepath = tmp_path / "test.nc"

    ds1 = xr.DataArray([1, 2, 3], name="foo").to_dataset()
    ds1.to_netcdf(filepath)
    ds2 = xr.DataArray([4, 5], name="bar").to_dataset()
    ds2.to_netcdf(filepath, group="subgroup", mode="a")

    return str(filepath)


@pytest.fixture
def netcdf4_files_factory(tmp_path: Path) -> Callable:
    def create_netcdf4_files(
        encoding: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[str, str]:
        filepath1 = tmp_path / "air1.nc"
        filepath2 = tmp_path / "air2.nc"

        with xr.tutorial.open_dataset("air_temperature") as ds:
            # Split dataset into two parts
            ds1 = ds.isel(time=slice(None, 1460))
            ds2 = ds.isel(time=slice(1460, None))

            # Save datasets to disk as NetCDF in the temporary directory with the provided encoding
            ds1.to_netcdf(filepath1, encoding=encoding)
            ds2.to_netcdf(filepath2, encoding=encoding)

        return str(filepath1), str(filepath2)

    return create_netcdf4_files


@pytest.fixture
def netcdf4_file_with_2d_coords(tmp_path: Path) -> str:
    filepath = tmp_path / "ROMS_example.nc"

    with xr.tutorial.open_dataset("ROMS_example") as ds:
        ds.to_netcdf(filepath, format="NETCDF4")

    return str(filepath)


@pytest.fixture
def netcdf4_virtual_dataset(netcdf4_file):
    from virtualizarr import open_virtual_dataset

    return open_virtual_dataset(netcdf4_file, indexes={})


@pytest.fixture
def netcdf4_inlined_ref(netcdf4_file):
    from kerchunk.hdf import SingleHdf5ToZarr

    return SingleHdf5ToZarr(netcdf4_file, inline_threshold=1000).translate()


@pytest.fixture
def hdf5_groups_file(tmp_path: Path) -> str:
    filepath = tmp_path / "air.nc"

    # Set up example xarray dataset
    with xr.tutorial.open_dataset("air_temperature") as ds:
        # Save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(filepath, format="NETCDF4", group="test/group")

    return str(filepath)


@pytest.fixture
def hdf5_empty(tmp_path: Path) -> str:
    filepath = tmp_path / "empty.nc"

    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("empty", shape=(), dtype="float32")
        dataset.attrs["empty"] = "true"

    return str(filepath)


@pytest.fixture
def hdf5_scalar(tmp_path: Path) -> str:
    filepath = tmp_path / "scalar.nc"

    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("scalar", data=0.1, dtype="float32")
        dataset.attrs["scalar"] = "true"

    return str(filepath)


@pytest.fixture
def simple_netcdf4(tmp_path: Path) -> str:
    filepath = tmp_path / "simple.nc"

    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4)
    var = Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    ds.to_netcdf(filepath)

    return str(filepath)


def offset_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) * 10


def length_from_chunk_key(ind: tuple[int, ...]) -> int:
    return sum(ind) + 5


def entry_from_chunk_key(ind: tuple[int, ...]) -> dict[str, str | int]:
    """Generate a (somewhat) unique manifest entry from a given chunk key"""
    entry = {
        "path": f"/foo.{str(join(ind))}.nc",
        "offset": offset_from_chunk_key(ind),
        "length": length_from_chunk_key(ind),
    }
    return entry  # type: ignore[return-value]


@pytest.fixture
def create_manifestarray(array_v3_metadata):
    """
    Create an example ManifestArray with sensible defaults.

    The manifest is populated with a (somewhat) unique path, offset, and length for each key.
    """

    def _create_manifestarray(
        shape: tuple | None = (5, 5),
        chunks: tuple | None = (5, 5),
        codecs: list[dict] | None = [
            {"configuration": {"endian": "little"}, "name": "bytes"},
            {"name": "numcodecs.zlib", "configuration": {"level": 1}},
        ],
    ):
        metadata = array_v3_metadata(shape=shape, chunks=chunks, codecs=codecs)
        chunk_grid_shape = tuple(
            ceildiv(axis_length, chunk_length)
            for axis_length, chunk_length in zip(shape, chunks)
        )

        if chunk_grid_shape == ():
            d = {"0": entry_from_chunk_key((0,))}
        else:
            # create every possible combination of keys
            all_possible_combos = itertools.product(
                *[range(length) for length in chunk_grid_shape]
            )
            d = {join(ind): entry_from_chunk_key(ind) for ind in all_possible_combos}

        chunkmanifest = ChunkManifest(entries=d)

        return ManifestArray(chunkmanifest=chunkmanifest, metadata=metadata)

    return _create_manifestarray


@pytest.fixture
def array_v3_metadata():
    def _create_metadata(
        shape: tuple = (5, 5),
        chunks: tuple = (5, 5),
        data_type: str = np.dtype("int32"),
        codecs: list[dict] | None = None,
        fill_value: int = None,
    ):
        codecs = codecs or [{"configuration": {"endian": "little"}, "name": "bytes"}]
        return create_v3_array_metadata(
            shape=shape,
            chunk_shape=chunks,
            data_type=data_type,
            codecs=codecs,
            fill_value=fill_value or 0,
        )

    return _create_metadata


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


@pytest.fixture
def gen_virtual_variable(array_v3_metadata: Callable) -> Callable:
    def _gen_virtual_variable(
        file_uri: str,
        shape: tuple[int, ...] = (3, 4),
        chunk_shape: tuple[int, ...] = (3, 4),
        dtype: np.dtype = np.dtype("int32"),
        codecs: Optional[list[dict[Any, Any]]] = None,
        fill_value: Optional[str] = None,
        encoding: Optional[dict] = None,
        offset: int = 6144,
        length: int = 48,
        dims: list[str] = [],
        attrs: dict[str, Any] = {},
    ) -> xr.Variable:
        manifest = generate_chunk_manifest(
            file_uri,
            shape=shape,
            chunks=chunk_shape,
            offset=offset,
            length=length,
        )
        metadata = array_v3_metadata(
            shape=shape,
            chunks=chunk_shape,
            codecs=codecs,
            data_type=dtype,
            fill_value=fill_value,
        )
        ma = ManifestArray(chunkmanifest=manifest, metadata=metadata)
        return xr.Variable(
            data=ma,
            dims=dims,
            encoding=encoding,
            attrs=attrs,
        )

    return _gen_virtual_variable


@pytest.fixture
def gen_virtual_dataset(gen_virtual_variable: Callable) -> Callable:
    def _gen_virtual_dataset(
        file_uri: str,
        shape: tuple[int, ...] = (3, 4),
        chunk_shape: tuple[int, ...] = (3, 4),
        dtype: np.dtype = np.dtype("int32"),
        codecs: Optional[list[dict[Any, Any]]] = None,
        fill_value: Optional[str] = None,
        encoding: Optional[dict] = None,
        variable_name: str = "foo",
        offset: int = 6144,
        length: int = 48,
        dims: Optional[list[str]] = None,
        coords: Optional[xr.Coordinates] = None,
    ) -> xr.Dataset:
        with xr.open_dataset(file_uri) as ds:
            var = gen_virtual_variable(
                file_uri=file_uri,
                shape=shape,
                chunk_shape=chunk_shape,
                dtype=dtype,
                codecs=codecs,
                fill_value=fill_value,
                encoding=encoding,
                offset=offset,
                length=length,
                dims=dims or [str(name) for name in ds.dims],
                attrs=ds[variable_name].attrs,
            )

            return xr.Dataset(
                {variable_name: var},
                coords=coords,
                attrs=ds.attrs,
            )

    return _gen_virtual_dataset


# Common codec configurations used across tests
@pytest.fixture
def delta_codec() -> Dict[str, Any]:
    """Delta codec configuration for array-to-array transformation."""
    return {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}}


@pytest.fixture
def arraybytes_codec() -> Dict[str, Any]:
    """Bytes codec configuration for array-to-bytes transformation."""
    return {"name": "bytes", "configuration": {"endian": "little"}}


@pytest.fixture
def blosc_codec() -> Dict[str, Any]:
    """Blosc codec configuration for bytes-to-bytes transformation."""
    return {
        "name": "blosc",
        "configuration": {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": "shuffle",
            "typesize": 4,
        },
    }


@pytest.fixture
def zlib_codec() -> Dict[str, Any]:
    """Zlib codec configuration for bytes-to-bytes transformation."""
    return {"name": "numcodecs.zlib", "configuration": {"level": 1}}
