import numpy as np
import pytest
from zarr.core.metadata.v3 import ArrayV3Metadata

from conftest import (
    ARRAYBYTES_CODEC,
    ZLIB_CODEC,
)
from virtualizarr.manifests import ChunkManifest, ManifestArray


class TestInit:
    def test_manifest_array(self, array_v3_metadata):
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        metadata = array_v3_metadata(shape=shape, chunks=chunks)

        marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        assert marr.chunks == chunks
        assert marr.dtype == np.dtype("int32")
        assert marr.shape == shape
        assert marr.size == 5 * 2 * 20
        assert marr.ndim == 3

    def test_manifest_array_dict_v3_metadata(self, array_v3_metadata):
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        metadata = array_v3_metadata(shape=shape, chunks=chunks)
        metadata_dict = ArrayV3Metadata.from_dict(metadata.to_dict())

        marr = ManifestArray(metadata=metadata_dict, chunkmanifest=manifest)
        assert marr.chunks == chunks
        assert marr.dtype == np.dtype("int32")
        assert marr.shape == shape
        assert marr.size == 5 * 2 * 20
        assert marr.ndim == 3


class TestResultType:
    def test_idempotent(self, manifest_array):
        marr1 = manifest_array(shape=(), chunks=(), data_type=np.dtype("int32"))
        marr2 = manifest_array(shape=(), chunks=(), data_type=np.dtype("int32"))

        assert np.result_type(marr1) == marr1.dtype
        assert np.result_type(marr1, marr1.dtype) == marr1.dtype
        assert np.result_type(marr1, marr2) == marr1.dtype

    def test_raises(self, manifest_array):
        marr1 = manifest_array(shape=(), chunks=(), data_type=np.dtype("int32"))
        marr2 = manifest_array(shape=(), chunks=(), data_type=np.dtype("int64"))

        with pytest.raises(ValueError, match="inconsistent"):
            np.result_type(marr1, marr2)


class TestEquals:
    def test_equals(self, array_v3_metadata):
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        metadata = array_v3_metadata(shape=shape, chunks=chunks)

        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest)
        result = marr1 == marr2
        assert isinstance(result, np.ndarray)
        assert result.shape == shape
        assert result.dtype == np.dtype(bool)
        assert result.all()

    def test_not_equal_chunk_entries(self, array_v3_metadata):
        # both manifest arrays in this example have the same metadata
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        metadata = array_v3_metadata(shape=shape, chunks=chunks)

        chunks_dict1 = {
            "0.0.0": {"path": "/oo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "/oo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0.0": {"path": "/oo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "/oo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)
        assert not (marr1 == marr2).all()

    @pytest.mark.skip(reason="Not Implemented")
    def test_partly_equals(self): ...

    def test_equals_nan_fill_value(self, array_v3_metadata):
        # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/501
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        metadata1 = array_v3_metadata(
            shape=(2,), chunks=(2,), data_type=np.float32, fill_value=np.float32("nan")
        )
        metadata2 = array_v3_metadata(
            shape=(2,), chunks=(2,), data_type=np.float32, fill_value=np.float32("nan")
        )
        marr1 = ManifestArray(metadata=metadata1, chunkmanifest=manifest)
        marr2 = ManifestArray(metadata=metadata2, chunkmanifest=manifest)

        result = marr1 == marr2
        assert result.all()


class TestBroadcast:
    def test_broadcast_existing_axis(self, manifest_array):
        marr = manifest_array(shape=(1, 2), chunks=(1, 2))
        expanded = np.broadcast_to(marr, shape=(3, 2))
        assert expanded.shape == (3, 2)
        assert expanded.chunks == (1, 2)
        assert expanded.manifest.dict() == {
            "0.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 8},
            "1.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 8},
            "2.0": {"path": "file:///foo.0.0.nc", "offset": 0, "length": 8},
        }

    def test_broadcast_new_axis(self, manifest_array):
        marr = manifest_array(shape=(3,), chunks=(1,))
        expanded = np.broadcast_to(marr, shape=(1, 3))
        assert expanded.shape == (1, 3)
        assert expanded.chunks == (1, 1)
        assert expanded.manifest.dict() == {
            "0.0": {"path": "file:///foo.0.nc", "offset": 0, "length": 4},
            "0.1": {"path": "file:///foo.1.nc", "offset": 0, "length": 4},
            "0.2": {"path": "file:///foo.2.nc", "offset": 0, "length": 4},
        }

    def test_broadcast_scalar(self, manifest_array):
        # regression test
        marr = manifest_array(shape=(), chunks=())
        assert marr.shape == ()
        assert marr.chunks == ()
        assert marr.manifest.dict() == {
            "0": {"path": "file:///foo.0.nc", "offset": 0, "length": 0},
        }

        expanded = np.broadcast_to(marr, shape=(1,))
        assert expanded.shape == (1,)
        assert expanded.chunks == (1,)
        assert expanded.manifest.dict() == {
            "0": {"path": "file:///foo.0.nc", "offset": 0, "length": 0},
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
    def test_raise_on_invalid_broadcast_shapes(
        self, shape, chunks, target_shape, manifest_array
    ):
        marr = manifest_array(shape=shape, chunks=chunks)
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
    def test_broadcast_any_shape(self, shape, chunks, target_shape, manifest_array):
        marr = manifest_array(shape=shape, chunks=chunks)

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
    def test_broadcast_empty(
        self, shape, chunks, grid_shape, target_shape, array_v3_metadata
    ):
        metadata = array_v3_metadata(chunks=chunks, shape=shape)
        manifest = ChunkManifest(entries={}, shape=grid_shape)
        marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)

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
    def test_concat(self, array_v3_metadata):
        # both manifest arrays in this example have the same metadata properties
        chunks_dict1 = {
            "0.0.0": {"path": "/foo1.nc", "offset": 100, "length": 100},
        }
        chunks_dict2 = {
            "0.0.0": {"path": "/foo2.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        manifest2 = ChunkManifest(entries=chunks_dict2)
        chunks = (5, 1, 10)
        shape = (5, 2, 20)
        metadata = array_v3_metadata(shape=shape, chunks=chunks)

        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)

        # Concatenate along the first axis
        concatenated = np.concatenate([marr1, marr2], axis=0)
        assert concatenated.shape == (10, 2, 20)
        assert concatenated.dtype == np.dtype("int32")

    def test_concat_empty(self, array_v3_metadata):
        chunks = (5, 1, 10)
        shape = (5, 1, 20)
        codecs = [ARRAYBYTES_CODEC, ZLIB_CODEC]
        metadata = array_v3_metadata(shape=shape, chunks=chunks, codecs=codecs)
        empty_chunks_dict = {}
        empty_chunk_manifest = ChunkManifest(entries=empty_chunks_dict, shape=(1, 1, 2))
        manifest_array_with_empty_chunks = ManifestArray(
            metadata=metadata, chunkmanifest=empty_chunk_manifest
        )

        chunks_dict = {
            "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        manifest_array = ManifestArray(metadata=metadata, chunkmanifest=manifest)

        # Concatenate with an empty array
        result = np.concatenate(
            [manifest_array_with_empty_chunks, manifest_array], axis=1
        )
        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        codec_dict = result.metadata.codecs[1].to_dict()
        assert codec_dict["name"] == "numcodecs.zlib"
        assert codec_dict["configuration"] == {"level": 1}
        assert result.metadata.fill_value == metadata.fill_value


class TestStack:
    def test_stack(self, array_v3_metadata):
        # both manifest arrays in this example have the same metadata
        chunks = (5, 10)
        shape = (5, 20)
        codecs = [ARRAYBYTES_CODEC, ZLIB_CODEC]
        metadata = array_v3_metadata(shape=shape, chunks=chunks, codecs=codecs)
        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        codec_dict = result.metadata.codecs[1].to_dict()
        assert codec_dict["name"] == "numcodecs.zlib"
        assert codec_dict["configuration"] == {"level": 1}
        assert result.metadata.fill_value == metadata.fill_value

    def test_stack_empty(self, array_v3_metadata):
        # both manifest arrays in this example have the same metadata properties
        chunks = (5, 10)
        shape = (5, 20)
        metadata = array_v3_metadata(
            shape=shape,
            chunks=chunks,
            codecs=[ARRAYBYTES_CODEC, ZLIB_CODEC],
        )

        chunks_dict1 = {}
        manifest1 = ChunkManifest(entries=chunks_dict1, shape=(1, 2))
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)

        result = np.stack([marr1, marr2], axis=1)

        assert result.shape == (5, 2, 20)
        assert result.chunks == (5, 1, 10)
        assert result.manifest.dict() == {
            "0.1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        codec_dict = result.metadata.codecs[1].to_dict()
        assert codec_dict["name"] == "numcodecs.zlib"
        assert result.metadata.fill_value == metadata.fill_value


def test_refuse_combine(array_v3_metadata):
    # TODO test refusing to concatenate arrays that have conflicting shapes / chunk sizes
    chunks = (5, 1, 10)
    shape = (5, 1, 20)
    metadata_common = array_v3_metadata(shape=shape, chunks=chunks)

    chunks_dict1 = {
        "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
    }
    chunkmanifest1 = ChunkManifest(entries=chunks_dict1)
    chunks_dict2 = {
        "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
    }
    chunkmanifest2 = ChunkManifest(entries=chunks_dict2)
    marr1 = ManifestArray(metadata=metadata_common, chunkmanifest=chunkmanifest1)

    metadata_different_codecs = array_v3_metadata(
        shape=shape,
        chunks=chunks,
        codecs=[ARRAYBYTES_CODEC, ZLIB_CODEC],
    )
    marr2 = ManifestArray(
        metadata=metadata_different_codecs, chunkmanifest=chunkmanifest2
    )
    for func in [np.concatenate, np.stack]:
        with pytest.raises(NotImplementedError, match="different codecs"):
            func([marr1, marr2], axis=0)

    metadata_wrong_dtype = array_v3_metadata(
        shape=shape, chunks=chunks, data_type=np.dtype("int64")
    )
    marr2 = ManifestArray(metadata=metadata_wrong_dtype, chunkmanifest=chunkmanifest2)
    for func in [np.concatenate, np.stack]:
        with pytest.raises(ValueError, match="inconsistent dtypes"):
            func([marr1, marr2], axis=0)

    metadata_variable_chunk = array_v3_metadata(
        shape=shape,
        chunks=(4, 1, 19),
    )
    marr = ManifestArray(metadata=metadata_variable_chunk, chunkmanifest=chunkmanifest2)
    with pytest.raises(
        ValueError,
        match="Cannot concatenate arrays with partial chunks because only regular chunk grids are currently supported. Concat input 0 has array length 5 along the concatenation axis which is not evenly divisible by chunk length 4.",
    ):
        np.concatenate([marr, marr], axis=0)


class TestIndexing:
    @pytest.mark.parametrize("dodgy_indexer", ["string", [], (5, "string")])
    def test_invalid_indexer_types(self, manifest_array, dodgy_indexer):
        marr = manifest_array(shape=(4,), chunks=(2,))

        with pytest.raises(TypeError, match="indexer must be of type"):
            marr[dodgy_indexer]

    @pytest.mark.parametrize(
        "in_shape, in_chunks, invalid_indexer",
        [
            ((2,), (1,), (0, 0)),
            ((2,), (1,), (0, None, 0)),
            ((2,), (1,), (0, ..., 0)),
            ((), (), (0)),
            # valid in numpy but not in the array API standard
            ((2,), (1,), None),
            ((2,), (1,), (None, None)),
        ],
    )
    def test_invalid_indexer_shape(
        self, manifest_array, in_shape, in_chunks, invalid_indexer
    ):
        marr = manifest_array(shape=in_shape, chunks=in_chunks)

        with pytest.raises(
            ValueError,
            match="Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions",
        ):
            marr[invalid_indexer]

    @pytest.mark.parametrize("unsupported_indexer", [np.ndarray(0)])
    def test_unsupported_index_types(self, manifest_array, unsupported_indexer):
        marr = manifest_array(shape=(2,), chunks=(1,))

        with pytest.raises(NotImplementedError):
            marr[unsupported_indexer]

    def test_indexing_scalar_with_ellipsis(self, manifest_array):
        # regression test for https://github.com/zarr-developers/VirtualiZarr/pull/641
        marr = manifest_array(shape=(), chunks=())
        assert marr[...] == marr

    def test_insert_newaxis_via_indexing_with_ellipsis(self, manifest_array):
        # regression test for GH issue #728
        marr = manifest_array(shape=(2,), chunks=(1,))
        new_marr = marr[None, ...]
        assert new_marr.shape == (1, 2)
        assert new_marr.chunks == (1, 1)

    @pytest.mark.parametrize(
        "in_shape, in_chunks, indexer, out_shape, out_chunks",
        [
            # no-ops
            ((2,), (1,), slice(None), (2,), (1,)),
            ((2,), (1,), ..., (2,), (1,)),
            ((2,), (1,), (..., slice(None)), (2,), (1,)),
            ((2,), (1,), (slice(None), ...), (2,), (1,)),
            ((), (), ..., (), ()),
            ((), (), (), (), ()),
            # inserting new axes
            ((2,), (1,), (None, slice(None)), (1, 2), (1, 1)),
            ((2,), (1,), (slice(None), None), (2, 1), (1, 1)),
            ((2,), (1,), (None, ...), (1, 2), (1, 1)),
            ((2,), (1,), (..., None), (2, 1), (1, 1)),
            ((), (), None, (1,), (1,)),
        ],
    )
    def test_noops_and_broadcasting_cases(
        self, manifest_array, in_shape, in_chunks, indexer, out_shape, out_chunks
    ):
        marr = manifest_array(shape=in_shape, chunks=in_chunks)
        indexed = marr[indexer]
        assert indexed.shape == out_shape
        assert indexed.chunks == out_chunks

    def test_raise_on_multiple_ellipses(
        self,
        manifest_array,
    ):
        marr = manifest_array(shape=(2,), chunks=(1,))
        with pytest.raises(ValueError, match="multiple Ellipses"):
            marr[..., ...]

    @pytest.mark.parametrize(
        "in_shape, in_chunks, indexer, out_shape, out_chunks",
        [
            # obvious no-ops
            ((2,), (1,), slice(0, 2), (2,), (1,)),
            # reduces shape
            pytest.param(
                (1,),
                (1,),
                0,
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            # requires chunk-aligned selection
            pytest.param(
                (2,),
                (1,),
                0,
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                1,
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                (0, ...),
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                (..., 0),
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                slice(0, 1),
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                (..., slice(0, 1)),
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
            pytest.param(
                (2,),
                (1,),
                (slice(0, 1), ...),
                (1,),
                (1,),
                marks=pytest.mark.xfail(
                    reason="Chunk-aligned indexing not yet implemented"
                ),
            ),
        ],
    )
    def test_chunk_selection_cases(
        self, manifest_array, in_shape, in_chunks, indexer, out_shape, out_chunks
    ):
        marr = manifest_array(shape=in_shape, chunks=in_chunks)
        indexed = marr[indexer]
        assert indexed.shape == out_shape
        assert indexed.chunks == out_chunks


def test_to_xarray(array_v3_metadata):
    chunks = (5, 10)
    shape = (5, 20)
    metadata = array_v3_metadata(
        shape=shape,
        chunks=chunks,
        attributes={"ham": "sandwich"},
        dimension_names=["x", "y"],
    )
    chunks_dict = {
        "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)

    vv = marr.to_virtual_variable()
    assert isinstance(vv.data, ManifestArray)
    assert vv.dims == ("x", "y")
    assert vv.data.metadata.dimension_names is None
    assert vv.attrs == {"ham": "sandwich"}
    assert vv.data.metadata.attributes == {}
