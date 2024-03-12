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


# TODO we really need some kind of fixtures to generate useful example data
# The hard part is having an alternative way to get to the expected result of concatenation
class TestConcat:
    def test_concat(self):
        chunks_dict1 = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        zarray1 = ZArray(
            chunks=(5, 1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )
        marr1 = ManifestArray(zarray=zarray1, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        zarray2 = ZArray(
            chunks=(5, 1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )
        marr2 = ManifestArray(zarray=zarray2, chunkmanifest=manifest2)

        result = np.concatenate([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }

    def test_refuse_concat(self):
        # TODO test refusing to concatenate arrays that have conflicting chunk sizes / codecs
        ...


class TestStack:
    def test_stack(self):
        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        zarray1 = ZArray(
            chunks=(5, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )
        marr1 = ManifestArray(zarray=zarray1, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        zarray2 = ZArray(
            chunks=(5, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )
        marr2 = ManifestArray(zarray=zarray2, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
