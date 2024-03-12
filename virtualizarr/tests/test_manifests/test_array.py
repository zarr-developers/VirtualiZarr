import numpy as np
import pytest

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.zarr import ZArray


class TestManifestArray:
    def test_create_manifestarray(self):
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        zarray = ZArray(
            chunks=chunks,
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )

        marr = ManifestArray(zarray=zarray, chunkmanifest=manifest)
        assert marr.chunks == chunks
        assert marr.dtype == np.dtype("int32")
        assert marr.shape == shape
        assert marr.size == 5 * 2 * 20
        assert marr.ndim == 3

    def test_create_invalid_manifestarray(self):
        chunks_dict = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        zarray = ZArray(
            chunks=chunks,
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )

        with pytest.raises(ValueError, match="Inconsistent chunk grid shape"):
            ManifestArray(zarray=zarray, chunkmanifest=manifest)

    def test_create_manifestarray_from_kerchunk_refs(self):
        arr_refs = {
            ".zarray": '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}',
            "0.0": ["test1.nc", 6144, 48],
        }
        marr = ManifestArray.from_kerchunk_refs(arr_refs)

        assert marr.shape == (2, 3)
        assert marr.chunks == (2, 3)
        assert marr.dtype == np.dtype("int64")
        assert marr.zarray.compressor is None
        assert marr.zarray.fill_value is None
        assert marr.zarray.filters is None
        assert marr.zarray.order == "C"


def test_concat():
    ...


def test_stack():
    ...
