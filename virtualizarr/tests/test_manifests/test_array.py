import numpy as np
import pytest

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import create_manifestarray
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
            compressor={"id": "zlib", "level": 1},
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
            compressor={"id": "zlib", "level": 1},
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0.0": {"path": "/oo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "/oo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "/oo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "/oo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        assert not (marr1 == marr2).all()

    @pytest.mark.skip(reason="Not Implemented")
    def test_partly_equals(self): ...


class TestBroadcast:
    def test_broadcast_existing_axis(self):
        marr = create_manifestarray(shape=(1, 2), chunks=(1, 2))
        expanded = np.broadcast_to(marr, shape=(3, 2))
        assert expanded.shape == (3, 2)
        assert expanded.chunks == (1, 2)
        assert expanded.manifest.dict() == {
            "0.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 5},
            "1.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 5},
            "2.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 5},
        }

    def test_broadcast_new_axis(self):
        marr = create_manifestarray(shape=(3,), chunks=(1,))
        expanded = np.broadcast_to(marr, shape=(1, 3))
        assert expanded.shape == (1, 3)
        assert expanded.chunks == (1, 1)
        assert expanded.manifest.dict() == {
            "0.0": {"path": "file:///foo.0.nc", "offset": 0, "length": 5},
            "0.1": {"path": "file:///foo.1.nc", "offset": 10, "length": 6},
            "0.2": {"path": "file:///foo.2.nc", "offset": 20, "length": 7},
        }

    def test_broadcast_scalar(self):
        # regression test
        marr = create_manifestarray(shape=(), chunks=())
        assert marr.shape == ()
        assert marr.chunks == ()
        assert marr.manifest.dict() == {
            "0": {"path": "file:///foo.0.nc", "offset": 0, "length": 5},
        }

        expanded = np.broadcast_to(marr, shape=(1,))
        assert expanded.shape == (1,)
        assert expanded.chunks == (1,)
        assert expanded.manifest.dict() == {
            "0": {"path": "file:///foo.0.nc", "offset": 0, "length": 5},
        }

    @pytest.mark.parametrize(
        "shape, chunks, target_shape",
        [
            ((1,), (1,), ()),
            ((2,), (1,), (1,)),
            ((3,), (2,), (5, 4, 4)),
            ((3, 2), (2, 2), (2, 3, 4)),
        ],
    )
    def test_raise_on_invalid_broadcast_shapes(self, shape, chunks, target_shape):
        marr = create_manifestarray(shape=shape, chunks=chunks)
        with pytest.raises(ValueError):
            np.broadcast_to(marr, shape=target_shape)

    # TODO replace this parametrization with hypothesis strategies
    @pytest.mark.parametrize(
        "shape, chunks, target_shape",
        [
            ((1,), (1,), (3,)),
            ((2,), (1,), (2,)),
            ((3,), (2,), (5, 4, 3)),
            ((3, 1), (2, 1), (2, 3, 4)),
        ],
    )
    def test_broadcast_any_shape(self, shape, chunks, target_shape):
        marr = create_manifestarray(shape=shape, chunks=chunks)

        # do the broadcasting
        broadcasted_marr = np.broadcast_to(marr, shape=target_shape)

        # check that the resultant shape is correct
        assert broadcasted_marr.shape == target_shape

        # check that chunk shape has plausible ndims and lengths
        broadcasted_chunk_shape = broadcasted_marr.chunks
        assert len(broadcasted_chunk_shape) == broadcasted_marr.ndim
        for len_arr, len_chunk in zip(broadcasted_marr.shape, broadcasted_chunk_shape):
            assert len_chunk <= len_arr

    @pytest.mark.parametrize(
        "shape, chunks, grid_shape, target_shape",
        [
            ((1,), (1,), (1,), (3,)),
            ((2,), (1,), (2,), (2,)),
            ((3,), (2,), (2,), (5, 4, 3)),
            ((3, 1), (2, 1), (2, 1), (2, 3, 4)),
        ],
    )
    def test_broadcast_empty(self, shape, chunks, grid_shape, target_shape):
        zarray = ZArray(
            chunks=chunks,
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )
        manifest = ChunkManifest(entries={}, shape=grid_shape)
        marr = ManifestArray(zarray, manifest)

        expanded = np.broadcast_to(marr, shape=target_shape)
        assert expanded.shape == target_shape
        assert len(expanded.chunks) == expanded.ndim
        assert all(
            len_chunk <= len_arr
            for len_arr, len_chunk in zip(expanded.shape, expanded.chunks)
        )
        assert expanded.manifest.dict() == {}


# TODO we really need some kind of fixtures to generate useful example data
# The hard part is having an alternative way to get to the expected result of concatenation
class TestConcat:
    def test_concat(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 1, 10),
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.concatenate([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        assert result.zarray.compressor == zarray.compressor
        assert result.zarray.filters == zarray.filters
        assert result.zarray.fill_value == zarray.fill_value
        assert result.zarray.order == zarray.order
        assert result.zarray.zarr_format == zarray.zarr_format

    def test_concat_empty(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 1, 10),
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {}
        manifest1 = ChunkManifest(entries=chunks_dict1, shape=(1, 1, 2))
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.concatenate([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        assert result.zarray.compressor == zarray.compressor
        assert result.zarray.filters == zarray.filters
        assert result.zarray.fill_value == zarray.fill_value
        assert result.zarray.order == zarray.order
        assert result.zarray.zarr_format == zarray.zarr_format

    def test_stack_empty(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 10),
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {}
        manifest1 = ChunkManifest(entries=chunks_dict1, shape=(1, 2))
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
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
        "compressor": {"id": "zlib", "level": 1},
        "dtype": np.dtype("int32"),
        "fill_value": 0.0,
        "filters": None,
        "order": "C",
        "shape": (5, 1, 10),
        "zarr_format": 2,
    }
    chunks_dict1 = {
        "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
    }
    chunkmanifest1 = ChunkManifest(entries=chunks_dict1)
    chunks_dict2 = {
        "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
    }
    chunkmanifest2 = ChunkManifest(entries=chunks_dict2)
    marr1 = ManifestArray(zarray=zarray_common, chunkmanifest=chunkmanifest1)

    zarray_wrong_compressor = zarray_common.copy()
    zarray_wrong_compressor["compressor"] = None
    marr2 = ManifestArray(zarray=zarray_wrong_compressor, chunkmanifest=chunkmanifest2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(NotImplementedError, match="different codecs"):
            func([marr1, marr2], axis=0)

    zarray_wrong_dtype = zarray_common.copy()
    zarray_wrong_dtype["dtype"] = np.dtype("int64")
    marr2 = ManifestArray(zarray=zarray_wrong_dtype, chunkmanifest=chunkmanifest2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(ValueError, match="inconsistent dtypes"):
            func([marr1, marr2], axis=0)

    zarray_wrong_dtype = zarray_common.copy()
    zarray_wrong_dtype["dtype"] = np.dtype("int64")
    marr2 = ManifestArray(zarray=zarray_wrong_dtype, chunkmanifest=chunkmanifest2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(ValueError, match="inconsistent dtypes"):
            func([marr1, marr2], axis=0)
