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
        marr = ManifestArray._from_kerchunk_refs(arr_refs)

        assert marr.shape == (2, 3)
        assert marr.chunks == (2, 3)
        assert marr.dtype == np.dtype("int64")
        assert marr.zarray.compressor is None
        assert marr.zarray.fill_value is None
        assert marr.zarray.filters is None
        assert marr.zarray.order == "C"


class TestEquals:
    def test_equals(self):
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

        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest)
        result = marr1 == marr2
        assert isinstance(result, np.ndarray)
        assert result.shape == shape
        assert result.dtype == np.dtype(bool)
        assert result.all()

    def test_not_equal_chunk_entries(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        assert not (marr1 == marr2).all()

    @pytest.mark.skip(reason="Not Implemented")
    def test_partly_equals(self):
        ...


# TODO we really need some kind of fixtures to generate useful example data
# The hard part is having an alternative way to get to the expected result of concatenation
class TestConcat:
    def test_concat(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.concatenate([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.zarray.compressor == zarray.compressor
        assert result.zarray.filters == zarray.filters
        assert result.zarray.fill_value == zarray.fill_value
        assert result.zarray.order == zarray.order
        assert result.zarray.zarr_format == zarray.zarr_format


class TestStack:
    def test_stack(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.zarray.compressor == zarray.compressor
        assert result.zarray.filters == zarray.filters
        assert result.zarray.fill_value == zarray.fill_value
        assert result.zarray.order == zarray.order
        assert result.zarray.zarr_format == zarray.zarr_format


def test_refuse_combine():
    # TODO test refusing to concatenate arrays that have conflicting shapes / chunk sizes

    zarray_common = {
        "chunks": (5, 1, 10),
        "compressor": "zlib",
        "dtype": np.dtype("int32"),
        "fill_value": 0.0,
        "filters": None,
        "order": "C",
        "shape": (5, 1, 10),
        "zarr_format": 2,
    }
    chunks_dict1 = {
        "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
    }
    chunks_dict2 = {
        "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
    }
    marr1 = ManifestArray(zarray=zarray_common, chunkmanifest=chunks_dict1)

    zarray_wrong_compressor = zarray_common.copy()
    zarray_wrong_compressor["compressor"] = None
    marr2 = ManifestArray(zarray=zarray_wrong_compressor, chunkmanifest=chunks_dict2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(NotImplementedError, match="different codecs"):
            func([marr1, marr2], axis=0)

    zarray_wrong_dtype = zarray_common.copy()
    zarray_wrong_dtype["dtype"] = np.dtype("int64")
    marr2 = ManifestArray(zarray=zarray_wrong_dtype, chunkmanifest=chunks_dict2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(ValueError, match="inconsistent dtypes"):
            func([marr1, marr2], axis=0)

    zarray_wrong_dtype = zarray_common.copy()
    zarray_wrong_dtype["dtype"] = np.dtype("int64")
    marr2 = ManifestArray(zarray=zarray_wrong_dtype, chunkmanifest=chunks_dict2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(ValueError, match="inconsistent dtypes"):
            func([marr1, marr2], axis=0)
