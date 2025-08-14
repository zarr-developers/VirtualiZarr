from pathlib import Path

import numcodecs
import numpy as np
import obstore
import pytest
import xarray as xr
from xarray import Dataset
from xarray.core.variable import Variable
from zarr.codecs import BytesCodec
from zarr.core.metadata import ArrayV3Metadata
from zarr.dtype import parse_data_type

from conftest import ARRAYBYTES_CODEC, ZLIB_CODEC
from virtualizarr.codecs import zarr_codec_config_to_v3
from virtualizarr.manifests import ChunkManifest, ManifestArray


@pytest.fixture
def vds_with_manifest_arrays(array_v3_metadata) -> Dataset:
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="/test.nc", offset=6144, length=48)}
        ),
        metadata=array_v3_metadata(
            shape=(2, 3),
            data_type=np.dtype("<i8"),
            chunks=(2, 3),
            codecs=[ARRAYBYTES_CODEC, ZLIB_CODEC],
            fill_value=0,
        ),
    )
    var = Variable(dims=["x", "y"], data=arr, attrs={"units": "km"})
    return Dataset({"a": var}, attrs={"something": 0})


@pytest.fixture()
def synthetic_vds(tmpdir: Path):
    filepath = f"{tmpdir}/data_chunk"
    store = obstore.store.LocalStore()
    arr = np.repeat([[1, 2]], 3, axis=1)
    shape = arr.shape
    dtype = arr.dtype
    buf = arr.tobytes()
    obstore.put(
        store,
        filepath,
        buf,
    )
    manifest = ChunkManifest(
        {"0.0": {"path": filepath, "offset": 0, "length": len(buf)}}
    )
    zdtype = parse_data_type(dtype, zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=[BytesCodec()],
        attributes={},
        dimension_names=("y", "x"),
        storage_transformers=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    foo = xr.Variable(data=ma, dims=["y", "x"], encoding={"scale_factor": 2})
    vds = xr.Dataset(
        {"foo": foo},
    )
    return vds, arr


@pytest.fixture()
def synthetic_vds_grid(tmpdir: Path):
    filepath = f"{tmpdir}/data_chunk"
    store = obstore.store.LocalStore()
    arr = np.repeat([[1, 2, 3, 4]], 2, axis=0)
    shape = arr.shape
    chunk_shape = (shape[0] // 2, shape[1] // 2)
    chunk_length = np.prod(chunk_shape) * arr.dtype.itemsize
    dtype = arr.dtype
    buf = arr.tobytes()
    obstore.put(
        store,
        filepath,
        buf,
    )
    manifest = ChunkManifest(
        {
            "0.0": {"path": filepath, "offset": 0, "length": chunk_length},
            "0.1": {"path": filepath, "offset": chunk_length, "length": chunk_length},
            "1.0": {
                "path": filepath,
                "offset": chunk_length * 2,
                "length": chunk_length,
            },
            "1.1": {
                "path": filepath,
                "offset": chunk_length * 3,
                "length": chunk_length,
            },
        }
    )
    zdtype = parse_data_type(dtype, zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=[BytesCodec()],
        attributes={},
        dimension_names=("y", "x"),
        storage_transformers=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    foo = xr.Variable(data=ma, dims=["y", "x"])
    vds = xr.Dataset(
        {"foo": foo},
    )
    return vds, arr


@pytest.fixture()
def compressed_synthetic_vds(tmpdir: Path):
    filepath = f"{tmpdir}/compressed_data_chunk"
    store = obstore.store.LocalStore()
    compressor = numcodecs.Zlib(level=9)
    arr = np.repeat([[1, 2]], 3, axis=1)
    dtype = arr.dtype
    shape = arr.shape
    compressed_buf = compressor.encode(arr.tobytes())
    obstore.put(
        store,
        filepath,
        compressed_buf,
    )
    manifest = ChunkManifest(
        {"0.0": {"path": filepath, "offset": 0, "length": len(compressed_buf)}}
    )
    zdtype = parse_data_type(dtype, zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=[BytesCodec(), zarr_codec_config_to_v3(compressor.get_config())],
        attributes={},
        dimension_names=("y", "x"),
        storage_transformers=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    foo = xr.Variable(data=ma, dims=["y", "x"])
    vds = xr.Dataset(
        {"foo": foo},
    )
    return vds, arr


@pytest.fixture()
def synthetic_vds_multiple_vars(synthetic_vds):
    return (
        xr.Dataset(
            {"foo": synthetic_vds[0]["foo"], "bar": synthetic_vds[0]["foo"]},
        ),
        synthetic_vds[1],
    )


@pytest.fixture()
def big_endian_synthetic_vds(tmpdir: Path):
    filepath = f"{tmpdir}/data_chunk"
    store = obstore.store.LocalStore()
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=">i4").reshape(3, 2)
    shape = arr.shape
    dtype = arr.dtype
    buf = arr.tobytes()
    obstore.put(
        store,
        filepath,
        buf,
    )
    manifest = ChunkManifest(
        {"0.0": {"path": filepath, "offset": 0, "length": len(buf)}}
    )
    zdtype = parse_data_type(dtype, zarr_format=3)
    metadata = ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=zdtype.default_scalar(),
        codecs=[BytesCodec(endian="big")],
        attributes={},
        dimension_names=("y", "x"),
        storage_transformers=None,
    )
    ma = ManifestArray(
        chunkmanifest=manifest,
        metadata=metadata,
    )
    foo = xr.Variable(data=ma, dims=["y", "x"], encoding={"scale_factor": 2})
    vds = xr.Dataset(
        {"foo": foo},
    )
    return vds, arr
