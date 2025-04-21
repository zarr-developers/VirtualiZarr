import h5py  # type: ignore
import numpy as np
import pytest
from obstore.store import LocalStore

from virtualizarr import open_virtual_dataset
from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetChunkManifest:
    def test_empty_chunks(self, empty_chunks_hdf5_file):
        f = h5py.File(empty_chunks_hdf5_file)
        ds = f["data"]
        manifest = HDFVirtualBackend._dataset_chunk_manifest(
            path=empty_chunks_hdf5_file, dataset=ds
        )
        assert manifest.shape_chunk_grid == (0,)

    def test_empty_dataset(self, empty_dataset_hdf5_file):
        f = h5py.File(empty_dataset_hdf5_file)
        ds = f["data"]
        manifest = HDFVirtualBackend._dataset_chunk_manifest(
            path=empty_dataset_hdf5_file, dataset=ds
        )
        assert manifest.shape_chunk_grid == (0,)

    def test_no_chunking(self, no_chunks_hdf5_file):
        f = h5py.File(no_chunks_hdf5_file)
        ds = f["data"]
        manifest = HDFVirtualBackend._dataset_chunk_manifest(
            path=no_chunks_hdf5_file, dataset=ds
        )
        assert manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_file):
        f = h5py.File(chunked_hdf5_file)
        ds = f["data"]
        manifest = HDFVirtualBackend._dataset_chunk_manifest(
            path=chunked_hdf5_file, dataset=ds
        )
        assert manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_file):
        f = h5py.File(chunked_roundtrip_hdf5_file)
        ds = f["var2"]
        manifest = HDFVirtualBackend._dataset_chunk_manifest(
            path=chunked_roundtrip_hdf5_file, dataset=ds
        )
        assert manifest.shape_chunk_grid == (2, 8)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_file):
        f = h5py.File(single_dimension_scale_hdf5_file)
        ds = f["data"]
        dims = HDFVirtualBackend._dataset_dims(ds)
        assert dims[0] == "x"

    def test_is_dimension_scale(self, is_scale_hdf5_file):
        f = h5py.File(is_scale_hdf5_file)
        ds = f["data"]
        dims = HDFVirtualBackend._dataset_dims(ds)
        assert dims[0] == "data"

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_file):
        f = h5py.File(multiple_dimension_scales_hdf5_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="dimension scales attached"):
            HDFVirtualBackend._dataset_dims(ds)

    def test_no_dimension_scales(self, no_chunks_hdf5_file):
        f = h5py.File(no_chunks_hdf5_file)
        ds = f["data"]
        dims = HDFVirtualBackend._dataset_dims(ds)
        assert dims == ["phony_dim_0", "phony_dim_1"]


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetToManifestArray:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_file):
        f = h5py.File(chunked_dimensions_netcdf4_file)
        ds = f["data"]
        ma = HDFVirtualBackend._construct_manifest_array(
            chunked_dimensions_netcdf4_file, ds, group=""
        )
        assert ma.chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_file):
        f = h5py.File(single_dimension_scale_hdf5_file)
        ds = f["data"]
        ma = HDFVirtualBackend._construct_manifest_array(
            single_dimension_scale_hdf5_file, ds, group=""
        )
        assert ma.chunks == (2,)

    def test_dataset_attributes(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        ma = HDFVirtualBackend._construct_manifest_array(
            string_attributes_hdf5_file, ds, group=""
        )
        assert ma.metadata.attributes["attribute_name"] == "attribute_name"

    def test_scalar_fill_value(self, scalar_fill_value_hdf5_file):
        f = h5py.File(scalar_fill_value_hdf5_file)
        ds = f["data"]
        ma = HDFVirtualBackend._construct_manifest_array(
            scalar_fill_value_hdf5_file, ds, group=""
        )
        assert ma.metadata.fill_value == 42

    def test_cf_fill_value(self, cf_fill_value_hdf5_file):
        f = h5py.File(cf_fill_value_hdf5_file)
        ds = f["data"]
        if ds.dtype.kind in "S":
            pytest.xfail("Investigate fixed-length binary encoding in Zarr v3")
        if ds.dtype.names:
            pytest.xfail(
                "To fix, structured dtype fill value encoding for Zarr backend"
            )
        ma = HDFVirtualBackend._construct_manifest_array(
            cf_fill_value_hdf5_file, ds, group=""
        )
        assert "_FillValue" in ma.metadata.attributes

    def test_cf_array_fill_value(self, cf_array_fill_value_hdf5_file):
        f = h5py.File(cf_array_fill_value_hdf5_file)
        ds = f["data"]
        ma = HDFVirtualBackend._construct_manifest_array(
            cf_array_fill_value_hdf5_file, ds, group=""
        )
        assert not isinstance(ma.metadata.attributes["_FillValue"], np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
class TestExtractAttributes:
    def test_string_attribute(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        attrs = HDFVirtualBackend._extract_attrs(ds)
        assert attrs["attribute_name"] == "attribute_name"

    def test_root_attribute(self, root_attributes_hdf5_file):
        f = h5py.File(root_attributes_hdf5_file)
        attrs = HDFVirtualBackend._extract_attrs(f)
        assert attrs["attribute_name"] == "attribute_name"

    def test_multiple_attributes(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        attrs = HDFVirtualBackend._extract_attrs(ds)
        assert len(attrs.keys()) == 2


@requires_hdf5plugin
@requires_imagecodecs
class TestManifestGroupFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_file):
        store = LocalStore()
        manifest_group = HDFVirtualBackend._construct_manifest_group(
            store=store,
            filepath=chunked_dimensions_netcdf4_file,
        )
        assert len(manifest_group.arrays) == 3

    def test_nested_groups_are_ignored(self, nested_group_hdf5_file):
        store = LocalStore()
        manifest_group = HDFVirtualBackend._construct_manifest_group(
            store=store,
            filepath=nested_group_hdf5_file,
            group="group",
        )
        assert len(manifest_group.arrays) == 1

    def test_drop_variables(self, multiple_datasets_hdf5_file):
        store = LocalStore()
        manifest_group = HDFVirtualBackend._construct_manifest_group(
            store=store,
            filepath=multiple_datasets_hdf5_file,
            drop_variables=["data2"],
        )
        assert "data2" not in manifest_group.arrays.keys()

    def test_dataset_in_group(self, group_hdf5_file):
        store = LocalStore()
        manifest_group = HDFVirtualBackend._construct_manifest_group(
            store=store,
            filepath=group_hdf5_file,
            group="group",
        )
        assert len(manifest_group.arrays) == 1

    def test_non_group_error(self, group_hdf5_file):
        store = LocalStore()
        with pytest.raises(ValueError):
            HDFVirtualBackend._construct_manifest_group(
                store=store,
                filepath=group_hdf5_file,
                group="group/data",
            )


@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualDataset:
    def test_coord_names(
        self,
        root_coordinates_hdf5_file,
    ):
        vds = HDFVirtualBackend.open_virtual_dataset(root_coordinates_hdf5_file)
        assert set(vds.coords) == {"lat", "lon"}

    @pytest.mark.xfail(reason="Requires Zarr v3 big endian dtype support")
    def test_big_endian(
        self,
        big_endian_dtype_hdf5_file,
    ):
        vds = HDFVirtualBackend.open_virtual_dataset(big_endian_dtype_hdf5_file)
        print(vds)


@requires_hdf5plugin
@requires_imagecodecs
@pytest.mark.parametrize("group", [None, "subgroup", "subgroup/"])
def test_subgroup_variable_names(netcdf4_file_with_data_in_multiple_groups, group):
    # regression test for GH issue #364
    vds = open_virtual_dataset(
        netcdf4_file_with_data_in_multiple_groups,
        group=group,
        backend=HDFVirtualBackend,
    )
    assert list(vds.dims) == ["dim_0"]
