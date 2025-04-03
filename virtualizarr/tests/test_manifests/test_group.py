import textwrap

import pytest
from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray, ManifestGroup


class TestManifestGroup:
    def test_group_containing_array(self, manifest_array):
        var = "foo"
        marr = manifest_array()
        manifest_group = ManifestGroup(arrays={var: marr}, attributes={})

        assert manifest_group.arrays == {var: marr}
        assert manifest_group.groups == {}
        assert isinstance(manifest_group[var], ManifestArray)
        with pytest.raises(KeyError):
            manifest_group["bar"]
        assert isinstance(manifest_group.metadata, GroupMetadata)
        assert len(manifest_group) == 1
        assert list(manifest_group) == [var]

    def test_manifest_repr(self, manifest_array):
        marr = manifest_array(shape=(5, 2), chunks=(5, 2))
        manifest_group = ManifestGroup(arrays={"foo": marr}, attributes={})
        expected_repr = textwrap.dedent(
            """
            ManifestGroup(
                arrays={'foo': ManifestArray<shape=(5, 2), dtype=int32, chunks=(5, 2)>},
                groups={},
                metadata=GroupMetadata(attributes={}, zarr_format=3, consolidated_metadata=None, node_type='group'),
            )
            """
        )
        assert repr(manifest_group) == expected_repr


class TestToXarray:
    def test_single_group_to_dataset(self, manifest_array):
        marr = manifest_array(shape=(3, 2), chunks=(3, 2), dimension_names=["x", "y"])
        manifest_group = ManifestGroup(arrays={"foo": marr}, attributes={"ham": "eggs"})

        vds = manifest_group.to_virtual_dataset()
        assert list(vds.variables) == ["foo"]
        assert vds.attrs == {"ham": "eggs"}
        assert list(vds.dims) == ["x", "y"]

        vv = vds.variables["foo"]
        assert isinstance(vv.data, ManifestArray)
        assert list(vv.dims) == ["x", "y"]
        # check dims info is not duplicated in two places
        assert vv.data.metadata.dimension_names is None

        # TODO test coordinates
