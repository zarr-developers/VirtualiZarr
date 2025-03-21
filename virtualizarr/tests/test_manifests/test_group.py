import pytest
from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup


@pytest.fixture
def manifest_array(array_v3_metadata):
    chunk_dict = {
        "0.0.0": {"path": "s3://bucket/foo.nc", "offset": 100, "length": 100},
        "0.0.1": {"path": "s3://bucket/foo.nc", "offset": 200, "length": 100},
        "0.1.0": {"path": "s3://bucket/foo.nc", "offset": 300, "length": 100},
        "0.1.1": {"path": "s3://bucket/foo.nc", "offset": 400, "length": 100},
    }
    manifest = ChunkManifest(entries=chunk_dict)
    chunks = (5, 1, 10)
    shape = (5, 2, 20)
    array_metadata = array_v3_metadata(shape=shape, chunks=chunks)
    return ManifestArray(metadata=array_metadata, chunkmanifest=manifest)


class TestManifestGroup:
    def test_manifest_array(self, array_v3_metadata, manifest_array):
        var = "foo"
        manifest_group = ManifestGroup(
            manifest_dict={var: manifest_array}, attributes={}
        )
        assert isinstance(manifest_group._manifest_dict, dict)
        assert isinstance(manifest_group._manifest_dict[var], ManifestArray)
        assert isinstance(manifest_group._metadata, GroupMetadata)

    def test_manifest_repr(self, manifest_array):
        manifest_group = ManifestGroup(
            manifest_dict={"foo": manifest_array}, attributes={}
        )
        assert str(manifest_group)
