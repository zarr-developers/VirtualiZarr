import textwrap

import pytest
from zarr.core.group import GroupMetadata

from virtualizarr.manifests import ManifestArray, ManifestGroup


class TestToVirtualDataset:
    def test_string_coordinates_attribute_not_substring_matched(self, manifest_array):
        # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/923
        # When group-level "coordinates" is a space-separated string, variables with
        # short names like "t" and "u" should NOT be incorrectly promoted to coordinates
        # (because "t" in "time step isobaricInhPa latitude longitude" is True as substring match)
        dims_5d = ["time", "step", "isobaricInhPa", "latitude", "longitude"]
        t_arr = manifest_array(
            shape=(1, 1, 1, 2, 2), chunks=(1, 1, 1, 2, 2), dimension_names=dims_5d
        )
        u_arr = manifest_array(
            shape=(1, 1, 1, 2, 2), chunks=(1, 1, 1, 2, 2), dimension_names=dims_5d
        )
        time_arr = manifest_array(shape=(1,), chunks=(1,), dimension_names=["time"])
        lat_arr = manifest_array(shape=(2,), chunks=(2,), dimension_names=["latitude"])

        manifest_group = ManifestGroup(
            arrays={"t": t_arr, "u": u_arr, "time": time_arr, "latitude": lat_arr},
            attributes={"coordinates": "time step isobaricInhPa latitude longitude"},
        )

        vds = manifest_group.to_virtual_dataset()

        assert "t" in vds.data_vars, "'t' should be a data variable, not a coordinate"
        assert "u" in vds.data_vars, "'u' should be a data variable, not a coordinate"
        assert "t" not in vds.coords, "'t' should not be a coordinate"
        assert "u" not in vds.coords, "'u' should not be a coordinate"
        assert "time" in vds.coords, "'time' should be a coordinate"
        assert "latitude" in vds.coords, "'latitude' should be a coordinate"


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
