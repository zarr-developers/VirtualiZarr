import numpy as np
import pytest
from pydantic import ValidationError

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    concat_manifests,
    stack_manifests,
)
from virtualizarr.zarr import ZArray


class TestCreateManifest:
    def test_create_manifest(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert manifest.dict() == chunks

        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert manifest.dict() == chunks

    def test_invalid_chunk_entries(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc"},
        }
        with pytest.raises(ValidationError, match="missing"):
            ChunkManifest(entries=chunks)

        chunks = {
            "0.0.0": {
                "path": "s3://bucket/foo.nc",
                "offset": "some nonsense",
                "length": 100,
            },
        }
        with pytest.raises(ValidationError, match="should be a valid integer"):
            ChunkManifest(entries=chunks)

    def test_invalid_chunk_keys(self):
        chunks = {
            "0.0.": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        with pytest.raises(ValueError, match="Invalid format for chunk key: '0.0.'"):
            ChunkManifest(entries=chunks)

        chunks = {
            "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
        }
        with pytest.raises(ValueError, match="Inconsistent number of dimensions"):
            ChunkManifest(entries=chunks)

        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
        }
        with pytest.raises(ValueError, match="do not form a complete grid"):
            ChunkManifest(entries=chunks)

        chunks = {
            "1": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        with pytest.raises(ValueError, match="do not form a complete grid"):
            ChunkManifest(entries=chunks)


class TestProperties:
    def test_chunk_grid_info(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert manifest.ndim_chunk_grid == 3
        assert manifest.shape_chunk_grid == (1, 2, 2)


# TODO could we use hypothesis to test this?
# Perhaps by testing the property that splitting along a dimension then concatenating the pieces along that dimension should recreate the original manifest?
class TestCombineManifests:
    def test_concat(self):
        manifest1 = ChunkManifest(
            entries={
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest(
            entries={
                "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )
        axis = 1
        expected = ChunkManifest(
            entries={
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
                "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )

        result = concat_manifests([manifest1, manifest2], axis=axis)
        assert result.dict() == expected.dict()

    def test_stack(self):
        manifest1 = ChunkManifest(
            entries={
                "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest(
            entries={
                "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )
        axis = 1
        expected = ChunkManifest(
            entries={
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
                "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )

        result = stack_manifests([manifest1, manifest2], axis=axis)
        assert result.dict() == expected.dict()


@pytest.mark.skip(reason="Not implemented")
class TestSerializeManifest:
    def test_serialize_manifest_to_zarr(self):
        ...

    def test_deserialize_manifest_from_zarr(self):
        ...


class TestManifestArray:
    def test_create_manifestarray(self):
        chunks_dict = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (10, 1, 5)
        zarray = ZArray(
            chunks=chunks,
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(100, 11, 20),
            zarr_format=2,
        )

        marr = ManifestArray(zarray=zarray, chunkmanifest=manifest)
        assert marr.chunks == chunks
        assert marr.dtype == np.dtype("int32")
        assert marr.shape == (100, 11, 20)
        assert marr.size == 100 * 11 * 20
        assert marr.ndim == 3

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
