import pickle

import numpy as np
import pytest

from virtualizarr.manifests import ChunkEntry, ChunkManifest


class TestPathValidation:
    def test_normalize_paths_to_uris(self):
        chunkentry = ChunkEntry.with_validation(
            path="/local/foo.nc",
            offset=100,
            length=100,
        )
        assert chunkentry["path"] == "file:///local/foo.nc"

    def test_only_allow_absolute_paths(self):
        with pytest.raises(ValueError, match="must be absolute"):
            ChunkEntry.with_validation(path="local/foo.nc", offset=100, length=100)

    def test_allow_empty_path(self):
        ChunkEntry.with_validation(
            path="",
            offset=100,
            length=100,
        )

    @pytest.mark.parametrize(
        "url", ["http://site.com/file.nc", "https://site.com/file.nc"]
    )
    def test_allow_http_urls(self, url):
        chunkentry = ChunkEntry.with_validation(path=url, offset=100, length=100)
        assert chunkentry["path"] == url

    @pytest.mark.parametrize(
        "path",
        [
            pytest.param(
                "s3://BUCKET/foo.nc",
                marks=pytest.mark.xfail(
                    reason="cloudpathlib should ideally do stricter validation - see https://github.com/drivendataorg/cloudpathlib/issues/489"
                ),
            ),
        ],
    )
    def test_catch_malformed_path(self, path):
        with pytest.raises(ValueError):
            ChunkEntry.with_validation(path=path, offset=100, length=100)

    def test_allow_path_containing_colon(self):
        # regression test for issue #593
        ChunkEntry.with_validation(
            path="file:///folder/wrfout_d01_2024-01-02_01:00:00.nc",
            offset=100,
            length=100,
        )


class TestConvertingRelativePathsUsingFSRoot:
    def test_fs_root_must_be_absolute(self):
        with pytest.raises(ValueError, match="fs_root must be an absolute"):
            ChunkEntry.with_validation(
                path="local/foo.nc", offset=100, length=100, fs_root="directory"
            )

    def test_fs_root_must_not_have_file_suffix(self):
        with pytest.raises(ValueError, match="fs_root must be an absolute"):
            ChunkEntry.with_validation(
                path="local/foo.nc", offset=100, length=100, fs_root="directory.nc"
            )

    @pytest.mark.parametrize(
        "fs_root, relative_path, expected_path",
        [
            (
                "file:///tom/home/",
                "directory/file.nc",
                "file:///tom/home/directory/file.nc",
            ),
            ("file:///tom/home/", "../file.nc", "file:///tom/file.nc"),
            pytest.param(
                "s3://bucket/",
                "directory/file.nc",
                "s3://bucket/directory/file.nc",
                marks=pytest.mark.xfail(
                    reason="passing an s3 url to fs_root is not yet implemented",
                ),
            ),
            pytest.param(
                "https://site.com/",
                "directory/file.nc",
                "https://site.com/directory/file.nc",
                marks=pytest.mark.xfail(
                    reason="passing a http url to fs_root is not yet implemented",
                ),
            ),
        ],
    )
    def test_convert_to_absolute_uri(self, fs_root, relative_path, expected_path):
        chunkentry = ChunkEntry.with_validation(
            path=relative_path, offset=100, length=100, fs_root=fs_root
        )
        assert chunkentry["path"] == expected_path


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

    def test_slash_separator(self):
        """Chunk keys using '/' separator should be accepted when separator='/'."""
        chunks = {
            "0/0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0/1": {"path": "/foo.nc", "offset": 200, "length": 100},
            "1/0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "1/1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks, separator="/")
        assert manifest.shape_chunk_grid == (2, 2)
        assert manifest.ndim_chunk_grid == 2

    def test_slash_separator_rejected_without_separator_param(self):
        """Chunk keys using '/' should fail validation when separator is '.' (the default)."""
        chunks = {
            "0/0": {"path": "/foo.nc", "offset": 100, "length": 100},
        }
        with pytest.raises(ValueError, match="Invalid format for chunk key"):
            ChunkManifest(entries=chunks)

    def test_remove_chunks_with_empty_paths(self):
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


class TestCreateFromArrays:
    def test_create_from_arrays(self):
        paths = np.asarray(["/foo1.nc", "/foo2.nc"], dtype=np.dtypes.StringDType)
        offsets = np.asarray([100, 200], dtype=np.uint64)
        lengths = np.asarray([100, 100], dtype=np.uint64)

        chunkmanifest = ChunkManifest.from_arrays(
            paths=paths, offsets=offsets, lengths=lengths, validate_paths=True
        )
        assert chunkmanifest.ndim_chunk_grid == 1
        assert chunkmanifest.shape_chunk_grid == (2,)
        expected_d = {
            "0": {"path": "file:///foo1.nc", "offset": 100, "length": 100},
            "1": {"path": "file:///foo2.nc", "offset": 200, "length": 100},
        }
        assert chunkmanifest.dict() == expected_d

    def test_validate_paths(self):
        bad_paths = np.asarray(["./foo.nc"], dtype=np.dtypes.StringDType)
        offsets = np.asarray([100], dtype=np.uint64)
        lengths = np.asarray([100], dtype=np.uint64)

        with pytest.raises(ValueError, match="must be absolute"):
            ChunkManifest.from_arrays(
                paths=bad_paths, offsets=offsets, lengths=lengths, validate_paths=True
            )

        # no error when validation skipped
        ChunkManifest.from_arrays(
            paths=bad_paths, offsets=offsets, lengths=lengths, validate_paths=False
        )


class TestEquals:
    def test_equals(self):
        manifest1 = ChunkManifest(
            entries={
                "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
                "0.0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
            }
        )
        manifest2 = ChunkManifest(
            entries={
                "0.0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
                "0.0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
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
            "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
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
            "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        with pytest.raises(TypeError):
            # list is an invalid arg type
            manifest.rename_paths(["file1.nc", "file2.nc"])

    def test_normalize_paths_to_uris(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        renamed = manifest.rename_paths("/home/directory/bar.nc")
        assert renamed.dict() == {
            "0.0.0": {
                "path": "file:///home/directory/bar.nc",
                "offset": 100,
                "length": 100,
            },
        }

    def test_catch_malformed_paths(self):
        chunks = {
            "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)

        with pytest.raises(ValueError):
            # list is an invalid arg type
            manifest.rename_paths("./foo.nc")


class TestNativeChunks:
    """Tests for native (in-memory) chunk support in ChunkManifest."""

    def test_create_manifest_with_native_chunks(self):
        chunks = {
            "0.0": {
                "path": "",
                "offset": 0,
                "length": 4,
                "data": b"\x00\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        assert len(manifest._native) == 1
        assert (0, 0) in manifest._native
        assert manifest._native[(0, 0)] == b"\x00\x01\x02\x03"

    def test_mixed_virtual_and_native(self):
        chunks = {
            "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.1": {
                "path": "",
                "offset": 0,
                "length": 3,
                "data": b"\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        assert len(manifest._native) == 1
        assert manifest._paths[(0, 0)] == "s3://bucket/foo.nc"
        assert manifest._paths[(0, 1)] == ""

    def test_dict_includes_native_data(self):
        chunks = {
            "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.1": {
                "path": "",
                "offset": 0,
                "length": 4,
                "data": b"\x00\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        d = manifest.dict()
        assert len(d) == 2
        assert d["0.0"] == {
            "path": "s3://bucket/foo.nc",
            "offset": 100,
            "length": 100,
        }
        assert d["0.1"]["data"] == b"\x00\x01\x02\x03"

    def test_getitem_native_chunk(self):
        chunks = {
            "0.0": {
                "path": "",
                "offset": 0,
                "length": 4,
                "data": b"\x00\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        entry = manifest["0.0"]
        assert "data" in entry
        assert entry["data"] == b"\x00\x01\x02\x03"
        assert entry["path"] == ""

    def test_nbytes_includes_native_data(self):
        chunks = {
            "0.0": {
                "path": "",
                "offset": 0,
                "length": 4,
                "data": b"\x00\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        # nbytes should include both the numpy arrays and the native data
        assert manifest.nbytes > 4

    def test_equals_with_native(self):
        chunks = {
            "0": {
                "path": "",
                "offset": 0,
                "length": 3,
                "data": b"\x01\x02\x03",
            },
        }
        m1 = ChunkManifest(entries=chunks)
        m2 = ChunkManifest(entries=chunks)
        assert m1 == m2

    def test_not_equals_different_native_data(self):
        m1 = ChunkManifest(
            entries={
                "0": {
                    "path": "",
                    "offset": 0,
                    "length": 3,
                    "data": b"\x01\x02\x03",
                },
            }
        )
        m2 = ChunkManifest(
            entries={
                "0": {
                    "path": "",
                    "offset": 0,
                    "length": 3,
                    "data": b"\x04\x05\x06",
                },
            }
        )
        assert m1 != m2

    def test_from_arrays_with_native(self):
        paths = np.asarray(["s3://bucket/foo.nc", ""], dtype=np.dtypes.StringDType)
        offsets = np.asarray([100, 0], dtype=np.uint64)
        lengths = np.asarray([100, 4], dtype=np.uint64)
        native = {(1,): b"\x00\x01\x02\x03"}
        manifest = ChunkManifest.from_arrays(
            paths=paths,
            offsets=offsets,
            lengths=lengths,
            validate_paths=False,
            native=native,
        )
        assert manifest._native == native
        entry = manifest["1"]
        assert "data" in entry
        assert entry["data"] == b"\x00\x01\x02\x03"

    def test_virtual_only_has_empty_native(self):
        chunks = {
            "0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert manifest._native == {}

    def test_repr_with_native(self):
        chunks = {
            "0": {
                "path": "",
                "offset": 0,
                "length": 3,
                "data": b"\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        assert "native_chunks=1" in repr(manifest)

    def test_repr_without_native(self):
        chunks = {
            "0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks)
        assert "native_chunks" not in repr(manifest)

    def test_rename_paths_preserves_native(self):
        chunks = {
            "0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "1": {
                "path": "",
                "offset": 0,
                "length": 3,
                "data": b"\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        renamed = manifest.rename_paths("s3://bucket/bar.nc")
        assert renamed._native == {(1,): b"\x01\x02\x03"}
        assert renamed.dict()["0"]["path"] == "s3://bucket/bar.nc"

    def test_pickle_roundtrip(self):
        chunks = {
            "0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
            "0.1": {
                "path": "",
                "offset": 0,
                "length": 4,
                "data": b"\x00\x01\x02\x03",
            },
        }
        manifest = ChunkManifest(entries=chunks)
        pickled = pickle.dumps(manifest)
        restored = pickle.loads(pickled)
        assert manifest == restored
        assert restored._native == {(0, 1): b"\x00\x01\x02\x03"}

    def test_chunk_entry_with_validation_native(self):
        entry = ChunkEntry.with_validation(
            path="", offset=0, length=0, data=b"\x01\x02\x03"
        )
        assert entry["data"] == b"\x01\x02\x03"
        assert entry["path"] == ""
        assert entry["length"] == 3
