import pytest

from virtualizarr.manifests import ChunkManifest


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

        chunks = {}
        manifest = ChunkManifest(entries=chunks, shape=(2, 2))
        assert manifest.dict() == chunks

    def test_create_manifest_empty_missing_shape(self):
        with pytest.raises(ValueError, match="chunk grid shape if no chunks"):
            ChunkManifest(entries={})

    def test_invalid_chunk_entries(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc"},
        }
        with pytest.raises(ValueError, match="must be of the form"):
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

    def test_empty_chunk_paths(self):
        chunks = {
            "0.0.0": {"path": "", "offset": 0, "length": 100},
            "1.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert len(manifest.dict()) == 1


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


class TestEquals:
    def test_equals(self):
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
        assert manifest1 != manifest2


@pytest.mark.skip(reason="Not implemented")
class TestSerializeManifest:
    def test_serialize_manifest_to_zarr(self): ...

    def test_deserialize_manifest_from_zarr(self): ...


class TestRenamePaths:
    def test_rename_to_str(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        renamed = manifest.rename_paths("s3://bucket/bar.nc")
        assert renamed.dict() == {
            "0.0.0": {"path": "s3://bucket/bar.nc", "offset": 100, "length": 100},
        }

    def test_rename_using_function(self):
        chunks = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        def local_to_s3_url(old_local_path: str) -> str:
            from pathlib import Path

            new_s3_bucket_url = "s3://bucket/"

            filename = Path(old_local_path).name
            return str(new_s3_bucket_url + filename)

        renamed = manifest.rename_paths(local_to_s3_url)
        assert renamed.dict() == {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }

    def test_invalid_type(self):
        chunks = {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        with pytest.raises(TypeError):
            manifest.rename_paths(["file1.nc", "file2.nc"])
