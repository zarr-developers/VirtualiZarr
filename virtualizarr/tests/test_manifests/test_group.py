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
        marr1 = manifest_array(
            shape=(3, 2, 5), chunks=(1, 2, 1), dimension_names=["x", "y", "t"]
        )
        marr2 = manifest_array(shape=(3, 2), chunks=(1, 2), dimension_names=["x", "y"])
        marr3 = manifest_array(shape=(5,), chunks=(5,), dimension_names=["t"])

        manifest_group = ManifestGroup(
            arrays={
                "T": marr1,  # data variable
                "elevation": marr2,  # 2D coordinate
                "t": marr3,  # 1D dimension coordinate
            },
            attributes={"coordinates": "elevation t", "ham": "eggs"},
        )

        vds = manifest_group.to_virtual_dataset()
        assert list(vds.variables) == ["T", "elevation", "t"]
        assert vds.attrs == {"ham": "eggs"}
        assert list(vds.dims) == ["x", "y", "t"]

        vv = vds.variables["T"]
        assert isinstance(vv.data, ManifestArray)
        assert list(vv.dims) == ["x", "y", "t"]
        # check dims info is not duplicated in two places
        assert vv.data.metadata.dimension_names is None
        assert vv.attrs == {}

        assert list(vds.coords) == ["elevation", "t"]
