"""Pytest configuration and fixtures for virtualizarr tests."""

# Standard library imports
import itertools
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

# Third-party imports
import h5py  # type: ignore[import]
import numpy as np
import pytest
import xarray as xr
from xarray.core.variable import Variable

# Local imports
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.utils import ceildiv


# Pytest configuration
def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )
    parser.addoption(
        "--run-minio-tests",
        action="store_true",
        help="runs tests requiring docker and minio",
    )


def pytest_runtest_setup(item):
    """Skip network tests unless explicitly enabled."""
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run tests requiring an internet connection"
        )
    if "minio" in item.keywords and not item.config.getoption("--run-minio-tests"):
        pytest.skip("set --run-minio-tests to run tests requiring docker and minio")


def _xarray_subset():
    ds = xr.tutorial.open_dataset("air_temperature")
    return ds.isel(time=slice(0, 10), lat=slice(0, 90), lon=slice(0, 180))


@pytest.fixture(params=[2, 3])
def zarr_store(tmpdir, request):
    ds = _xarray_subset()
    filepath = f"{tmpdir}/air.zarr"
    ds.to_zarr(filepath, zarr_format=request.param)
    ds.close()
    return filepath


@pytest.fixture()
def zarr_store_scalar(tmpdir):
    import zarr

    store = zarr.storage.MemoryStore()
    zarr_store_scalar = zarr.create_array(store=store, shape=(), dtype="int8")
    zarr_store_scalar[()] = 42
    return zarr_store_scalar


# Common codec configurations
DELTA_CODEC = {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}}
ARRAYBYTES_CODEC = {"name": "bytes", "configuration": {"endian": "little"}}
BLOSC_CODEC = {
    "name": "blosc",
    "configuration": {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "shuffle",
        "typesize": 4,
    },
}
ZLIB_CODEC = {"name": "numcodecs.zlib", "configuration": {"level": 1}}


# Helper functions
def _generate_chunk_entries(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    entry_generator: Callable[[tuple[int, ...]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Generate chunk entries for a manifest based on shape and chunks.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array
    chunks : tuple of int
        The chunk size for each dimension
    entry_generator : callable
        Function that takes chunk indices and returns an entry dict

    Returns
    -------
    dict
        Mapping of chunk keys to entry dictionaries
    """
    chunk_grid_shape = tuple(
        ceildiv(axis_length, chunk_length)
        for axis_length, chunk_length in zip(shape, chunks)
    )

    if chunk_grid_shape == ():
        return {"0": entry_generator((0,))}

    all_possible_combos = itertools.product(
        *[range(length) for length in chunk_grid_shape]
    )
    return {join(ind): entry_generator(ind) for ind in all_possible_combos}


def _offset_from_chunk_key(ind: tuple[int, ...]) -> int:
    """Generate an offset value from chunk indices."""
    return sum(ind) * 10


def _length_from_chunk_key(ind: tuple[int, ...]) -> int:
    """Generate a length value from chunk indices."""
    return sum(ind) + 5


def _entry_from_chunk_key(ind: tuple[int, ...]) -> dict[str, str | int]:
    """Generate a (somewhat) unique manifest entry from a given chunk key."""
    entry = {
        "path": f"/foo.{str(join(ind))}.nc",
        "offset": _offset_from_chunk_key(ind),
        "length": _length_from_chunk_key(ind),
    }
    return entry  # type: ignore[return-value]


def _generate_chunk_manifest(
    netcdf4_file: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    offset: int = 6144,
    length: int = 48,
) -> ChunkManifest:
    """Generate a chunk manifest with sequential offsets for each chunk."""
    current_offset = [offset]  # Use list to allow mutation in closure

    def sequential_entry_generator(ind: tuple[int, ...]) -> dict[str, Any]:
        entry = {
            "path": netcdf4_file,
            "offset": current_offset[0],
            "length": length,
        }
        current_offset[0] += length
        return entry

    entries = _generate_chunk_entries(shape, chunks, sequential_entry_generator)
    return ChunkManifest(entries)


# NetCDF file fixtures
@pytest.fixture
def empty_netcdf4_file(tmp_path: Path) -> str:
    """Create an empty NetCDF4 file."""
    filepath = tmp_path / "empty.nc"
    with xr.Dataset() as ds:
        ds.to_netcdf(filepath, format="NETCDF4")
    return str(filepath)


@pytest.fixture
def netcdf4_file(tmp_path: Path) -> str:
    """Create a NetCDF4 file with air temperature data."""
    filepath = tmp_path / "air.nc"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.to_netcdf(filepath, format="NETCDF4")
    return str(filepath)


@pytest.fixture
def netcdf4_file_with_data_in_multiple_groups(tmp_path: Path) -> str:
    """Create a NetCDF4 file with data in multiple groups."""
    filepath = tmp_path / "test.nc"
    ds1 = xr.DataArray([1, 2, 3], name="foo").to_dataset()
    ds1.to_netcdf(filepath)
    ds2 = xr.DataArray([4, 5], name="bar").to_dataset()
    ds2.to_netcdf(filepath, group="subgroup", mode="a")
    return str(filepath)


@pytest.fixture
def netcdf4_files_factory(tmp_path: Path) -> Callable[[], tuple[str, str]]:
    """Factory fixture to create multiple NetCDF4 files."""

    def create_netcdf4_files(
        encoding: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[str, str]:
        filepath1 = tmp_path / "air1.nc"
        filepath2 = tmp_path / "air2.nc"

        with xr.tutorial.open_dataset("air_temperature") as ds:
            ds1 = ds.isel(time=slice(None, 1460))
            ds2 = ds.isel(time=slice(1460, None))

            ds1.to_netcdf(filepath1, encoding=encoding)
            ds2.to_netcdf(filepath2, encoding=encoding)

        return str(filepath1), str(filepath2)

    return create_netcdf4_files


@pytest.fixture
def netcdf4_files_factory_2d(tmp_path: Path) -> Callable[[], tuple[str, str, str, str]]:
    """Factory fixture to create multiple NetCDF4 files."""

    def create_netcdf4_files(
        encoding: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> tuple[str, str, str, str]:
        filepath1 = tmp_path / "air1.nc"
        filepath2 = tmp_path / "air2.nc"
        filepath3 = tmp_path / "air3.nc"
        filepath4 = tmp_path / "air4.nc"

        with xr.tutorial.open_dataset("air_temperature") as ds:
            ds1 = ds.isel(time=slice(None, 1460), lat=slice(None, 10))
            ds2 = ds.isel(time=slice(1460, None), lat=slice(None, 10))
            ds3 = ds.isel(time=slice(None, 1460), lat=slice(10, 20))
            ds4 = ds.isel(time=slice(1460, None), lat=slice(10, 20))

            ds1.to_netcdf(filepath1, encoding=encoding)
            ds2.to_netcdf(filepath2, encoding=encoding)
            ds3.to_netcdf(filepath3, encoding=encoding)
            ds4.to_netcdf(filepath4, encoding=encoding)

        return str(filepath1), str(filepath2), str(filepath3), str(filepath4)

    return create_netcdf4_files


@pytest.fixture
def netcdf4_file_with_2d_coords(tmp_path: Path) -> str:
    """Create a NetCDF4 file with 2D coordinates."""
    filepath = tmp_path / "ROMS_example.nc"
    with xr.tutorial.open_dataset("ROMS_example") as ds:
        ds.to_netcdf(filepath, format="NETCDF4")
    return str(filepath)


@pytest.fixture
def netcdf4_virtual_dataset(netcdf4_file):
    """Create a virtual dataset from a NetCDF4 file."""
    from virtualizarr import open_virtual_dataset

    with open_virtual_dataset(netcdf4_file, loadable_variables=[]) as ds:
        yield ds


@pytest.fixture
def netcdf4_inlined_ref(netcdf4_file):
    """Create an inlined reference from a NetCDF4 file."""
    from kerchunk.hdf import SingleHdf5ToZarr

    return SingleHdf5ToZarr(netcdf4_file, inline_threshold=1000).translate()


# HDF5 file fixtures
@pytest.fixture
def hdf5_groups_file(tmp_path: Path) -> str:
    """Create an HDF5 file with groups."""
    filepath = tmp_path / "air.nc"
    with xr.tutorial.open_dataset("air_temperature") as ds:
        ds.to_netcdf(filepath, format="NETCDF4", group="test/group")
    return str(filepath)


@pytest.fixture
def hdf5_empty(tmp_path: Path) -> str:
    """Create an empty HDF5 file."""
    filepath = tmp_path / "empty.nc"
    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("empty", shape=(), dtype="float32")
        dataset.attrs["empty"] = "true"
    return str(filepath)


@pytest.fixture
def hdf5_scalar(tmp_path: Path) -> str:
    """Create an HDF5 file with a scalar dataset."""
    filepath = tmp_path / "scalar.nc"
    with h5py.File(filepath, "w") as f:
        dataset = f.create_dataset("scalar", data=0.1, dtype="float32")
        dataset.attrs["scalar"] = "true"
    return str(filepath)


@pytest.fixture
def simple_netcdf4(tmp_path: Path) -> str:
    """Create a simple NetCDF4 file with a single variable."""
    filepath = tmp_path / "simple.nc"
    arr = np.arange(12, dtype=np.dtype("int32")).reshape(3, 4)
    var = Variable(data=arr, dims=["x", "y"])
    ds = xr.Dataset({"foo": var})
    ds.to_netcdf(filepath)
    return str(filepath)


# Zarr ArrayV3Metadata, ManifestArray, virtual xr.Variable and virtual xr.Dataset fixtures
@pytest.fixture
def array_v3_metadata():
    """Create V3 array metadata with sensible defaults."""

    def _create_metadata(
        shape: tuple = (5, 5),
        chunks: tuple = (5, 5),
        data_type: np.dtype = np.dtype("int32"),
        codecs: list[dict] | None = None,
        fill_value: int | float | None = None,
        attributes: dict | None = None,
        dimension_names: Iterable[str] | None = None,
    ):
        codecs = codecs or [{"configuration": {"endian": "little"}, "name": "bytes"}]
        return create_v3_array_metadata(
            shape=shape,
            chunk_shape=chunks,
            data_type=data_type,
            codecs=codecs,
            fill_value=fill_value or 0,
            attributes=attributes,
            dimension_names=dimension_names,
        )

    return _create_metadata


@pytest.fixture
def manifest_array(array_v3_metadata):
    """
    Create an example ManifestArray with sensible defaults.

    The manifest is populated with a (somewhat) unique path, offset, and length for each key.
    """

    def _manifest_array(
        shape: tuple = (5, 2),
        chunks: tuple = (5, 2),
        codecs: list[dict] | None = [ARRAYBYTES_CODEC, ZLIB_CODEC],
        dimension_names: Iterable[str] | None = None,
    ):
        metadata = array_v3_metadata(
            shape=shape, chunks=chunks, codecs=codecs, dimension_names=dimension_names
        )
        entries = _generate_chunk_entries(shape, chunks, _entry_from_chunk_key)
        chunkmanifest = ChunkManifest(entries=entries)
        return ManifestArray(chunkmanifest=chunkmanifest, metadata=metadata)

    return _manifest_array


@pytest.fixture
def virtual_variable(array_v3_metadata: Callable) -> Callable:
    """Generate a virtual variable with configurable parameters."""

    def _virtual_variable(
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
        manifest = _generate_chunk_manifest(
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

    return _virtual_variable


@pytest.fixture
def virtual_dataset(virtual_variable: Callable) -> Callable:
    """Generate a virtual dataset with configurable parameters."""

    def _virtual_dataset(
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
            var = virtual_variable(
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

    return _virtual_dataset


# Zarr fixtures
@pytest.fixture
def zarr_array():
    def create_zarr_array(codecs=None, zarr_format=3):
        """Create a test Zarr array with the specified codecs."""
        import zarr

        # Create a Zarr array in memory with the codecs
        zarr_array = zarr.create(
            shape=(1000, 1000),
            chunks=(100, 100),
            dtype="int32",
            store=None,
            zarr_format=zarr_format,
            codecs=codecs,
        )

        # Populate the Zarr array with data
        zarr_array[:] = np.arange(1000 * 1000).reshape(1000, 1000)
        return zarr_array

    return create_zarr_array
