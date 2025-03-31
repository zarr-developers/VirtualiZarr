import textwrap

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
    def test_group_containing_array(self, manifest_array):
        var = "foo"
        manifest_group = ManifestGroup(arrays={var: manifest_array}, attributes={})

        assert manifest_group.arrays == {var: manifest_array}
        assert manifest_group.groups == {}
        assert isinstance(manifest_group[var], ManifestArray)
        with pytest.raises(KeyError):
            manifest_group["bar"]
        assert isinstance(manifest_group.metadata, GroupMetadata)
        assert len(manifest_group) == 1
        assert list(manifest_group) == [var]

    def test_manifest_repr(self, manifest_array):
        manifest_group = ManifestGroup(arrays={"foo": manifest_array}, attributes={})
        expected_repr = textwrap.dedent(
            """
            ManifestGroup(
                arrays={'foo': ManifestArray<shape=(5, 2, 20), dtype=int32, chunks=(5, 1, 10)>},
                groups={},
                metadata=GroupMetadata(attributes={}, zarr_format=3, consolidated_metadata=None, node_type='group'),
            )
            """
        )
        assert repr(manifest_group) == expected_repr
