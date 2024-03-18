import numpy as np
import pytest

from virtualizarr.manifests import ChunkManifest


class TestCreateManifest:
    def test_create_manifest(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest.from_dict(chunks)
        assert manifest.dict() == chunks

        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest.from_dict(chunks)
        assert manifest.dict() == chunks

    def test_invalid_chunk_entries(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc"},
        }
        with pytest.raises(ValueError, match="must be of the form"):
            ChunkManifest.from_dict(chunks)

        chunks = {
            "0.0.0": {
                "path": "s3://bucket/foo.nc",
                "offset": "some nonsense",
                "length": 100,
            },
        }
        with pytest.raises(ValueError, match="must be of the form"):
            ChunkManifest.from_dict(chunks)

    def test_invalid_chunk_keys(self):
        chunks = {
            "0.0.": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        with pytest.raises(ValueError, match="Invalid format for chunk key: '0.0.'"):
            ChunkManifest.from_dict(chunks)

        chunks = {
            "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
        }
        with pytest.raises(ValueError, match="Inconsistent number of dimensions"):
            ChunkManifest.from_dict(chunks)


class TestProperties:
    def test_chunk_grid_info(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
            "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
            "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest.from_dict(chunks)
        assert manifest.ndim_chunk_grid == 3
        assert manifest.shape_chunk_grid == (1, 2, 2)


class TestEquals:
    def test_equals(self):
        manifest1 = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )
        assert manifest1 != manifest2


# TODO could we use hypothesis to test this?
# Perhaps by testing the property that splitting along a dimension then concatenating the pieces along that dimension should recreate the original manifest?
class TestCombineManifests:
    def test_concat(self):
        manifest1 = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )
        axis = 1
        expected = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
                "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )

        result_manifest = np.concatenate(
            [manifest1.entries, manifest2.entries], axis=axis
        )
        result = ChunkManifest(entries=result_manifest)
        assert result.dict() == expected.dict()

    def test_stack(self):
        manifest1 = ChunkManifest.from_dict(
            {
                "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest.from_dict(
            {
                "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )
        axis = 1
        expected = ChunkManifest.from_dict(
            {
                "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
                "0.1.0": {"path": "foo.nc", "offset": 300, "length": 100},
                "0.1.1": {"path": "foo.nc", "offset": 400, "length": 100},
            }
        )

        result_manifest = np.stack([manifest1.entries, manifest2.entries], axis=axis)
        result = ChunkManifest(entries=result_manifest)
        assert result.dict() == expected.dict()


@pytest.mark.skip(reason="Not implemented")
class TestSerializeManifest:
    def test_serialize_manifest_to_zarr(self):
        ...

    def test_deserialize_manifest_from_zarr(self):
        ...
