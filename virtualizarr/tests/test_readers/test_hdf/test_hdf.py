import h5py  # type: ignore
import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.backends.hdf import HDFBackend
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)
from virtualizarr.tests.utils import obstore_local


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetChunkManifest:
    @pytest.mark.xfail(reason="Tutorial data non coord dimensions are serialized with big endidan types and internally dropped")
    def test_empty_chunks(self, empty_chunks_hdf5_file):
        store = obstore_local(filepath=empty_chunks_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=empty_chunks_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_empty_dataset(self, empty_dataset_hdf5_file):
        store = obstore_local(filepath=empty_dataset_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=empty_dataset_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_no_chunking(self, no_chunks_hdf5_file):
        store = obstore_local(filepath=no_chunks_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=no_chunks_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_file):
        store = obstore_local(filepath=chunked_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=chunked_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_file):
        store = obstore_local(filepath=chunked_roundtrip_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=chunked_roundtrip_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["var2"].manifest.shape_chunk_grid == (2, 8)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_file):
        store = obstore_local(filepath=single_dimension_scale_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=single_dimension_scale_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("x",)

    def test_is_dimension_scale(self, is_scale_hdf5_file):
        store = obstore_local(filepath=is_scale_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=is_scale_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("data",)

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_file):
        store = obstore_local(filepath=multiple_dimension_scales_hdf5_file) 
        backend = HDFBackend()
        with pytest.raises(ValueError, match="dimension scales attached"):
            backend(
                filepath=multiple_dimension_scales_hdf5_file,
                object_reader=store
            )

    def test_no_dimension_scales(self, no_chunks_hdf5_file):
        store = obstore_local(filepath=no_chunks_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=no_chunks_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("phony_dim_0", "phony_dim_1")


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetToManifestArray:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_file):
        store = obstore_local(filepath=chunked_dimensions_netcdf4_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=chunked_dimensions_netcdf4_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_file):
        store = obstore_local(filepath=single_dimension_scale_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=single_dimension_scale_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.arrays["data"].chunks == (2,)

    def test_dataset_attributes(self, string_attributes_hdf5_file):
        store = obstore_local(filepath=string_attributes_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=string_attributes_hdf5_file,
            object_reader=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.attributes["attribute_name"] == "attribute_name"

    def test_scalar_fill_value(self, scalar_fill_value_hdf5_file):
        store = obstore_local(filepath=scalar_fill_value_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=scalar_fill_value_hdf5_file,
            object_reader=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.fill_value == 42

    def test_cf_fill_value(self, cf_fill_value_hdf5_file):
        f = h5py.File(cf_fill_value_hdf5_file)
        ds = f["data"]
        if ds.dtype.kind in "S":
            pytest.xfail("Investigate fixed-length binary encoding in Zarr v3")
        if ds.dtype.names:
            pytest.xfail(
                "To fix, structured dtype fill value encoding for Zarr backend"
            )
        store = obstore_local(filepath=cf_fill_value_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=cf_fill_value_hdf5_file,
            object_reader=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes

    def test_cf_array_fill_value(self, cf_array_fill_value_hdf5_file):
        store = obstore_local(filepath=cf_array_fill_value_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=cf_array_fill_value_hdf5_file,
            object_reader=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert not isinstance(metadata.attributes["_FillValue"], np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
class TestExtractAttributes:
    def test_root_attribute(self, root_attributes_hdf5_file):
        store = obstore_local(filepath=root_attributes_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=root_attributes_hdf5_file,
            object_reader=store
        )
        assert manifest_store._group.metadata.attributes["attribute_name"] == "attribute_name"
        
    def test_multiple_attributes(self, string_attributes_hdf5_file):
        store = obstore_local(filepath=string_attributes_hdf5_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=string_attributes_hdf5_file,
            object_reader=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert len(metadata.attributes.keys()) == 2

@requires_hdf5plugin
@requires_imagecodecs
class TestManifestGroupFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_file):
        store = obstore_local(filepath=chunked_dimensions_netcdf4_file) 
        backend = HDFBackend()
        manifest_store = backend(
            filepath=chunked_dimensions_netcdf4_file,
            object_reader=store
        )
        assert len(manifest_store._group.arrays) == 3

    def test_nested_groups_are_ignored(self, nested_group_hdf5_file):
        store = obstore_local(filepath=nested_group_hdf5_file) 
        backend = HDFBackend(group="group")
        manifest_store = backend(
            filepath=nested_group_hdf5_file,
            object_reader=store
        )
        assert len(manifest_store._group.arrays) == 1

    def test_drop_variables(self, multiple_datasets_hdf5_file):
        store = obstore_local(filepath=multiple_datasets_hdf5_file) 
        backend = HDFBackend(drop_variables=["data2"])
        manifest_store = backend(
            filepath=multiple_datasets_hdf5_file,
            object_reader=store
        )
        assert "data2" not in manifest_store._group.arrays.keys()

    def test_dataset_in_group(self, group_hdf5_file):
        store = obstore_local(filepath=group_hdf5_file) 
        backend = HDFBackend(group="group")
        manifest_store = backend(
            filepath=group_hdf5_file,
            object_reader=store
        )
        assert len(manifest_store._group.arrays) == 1

    def test_non_group_error(self, group_hdf5_file):
        store = obstore_local(filepath=group_hdf5_file) 
        backend = HDFBackend(group="group/data")
        with pytest.raises(ValueError):
            backend(
                filepath=group_hdf5_file,
                object_reader=store
            )

@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualDataset:
    def test_coord_names(
        self,
        root_coordinates_hdf5_file,
    ):
        store = obstore_local(filepath=root_coordinates_hdf5_file) 
        backend = HDFBackend()
        vds = open_virtual_dataset(
            filepath=root_coordinates_hdf5_file,
            object_reader=store,
            backend=backend,
        )
        assert set(vds.coords) == {"lat", "lon"}

    @pytest.mark.xfail(reason="Requires Zarr v3 big endian dtype support")
    def test_big_endian(
        self,
        big_endian_dtype_hdf5_file,
    ):
        store = obstore_local(filepath=big_endian_dtype_hdf5_file) 
        backend = HDFBackend()
        vds = open_virtual_dataset(
            filepath=big_endian_dtype_hdf5_file,
            object_reader=store,
            backend=backend,
        )
        print(vds)


@requires_hdf5plugin
@requires_imagecodecs
@pytest.mark.parametrize("group", [None, "subgroup", "subgroup/"])
def test_subgroup_variable_names(netcdf4_file_with_data_in_multiple_groups, group):
    # regression test for GH issue #364
    store = obstore_local(filepath=netcdf4_file_with_data_in_multiple_groups) 
    backend = HDFBackend(group=group)
    vds = open_virtual_dataset(
        filepath=netcdf4_file_with_data_in_multiple_groups,
        object_reader=store,
        backend=backend,
    )
    assert list(vds.dims) == ["dim_0"]
