import h5py  # type: ignore
import numpy as np
import pytest

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)
from virtualizarr.tests.utils import obstore_local


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetChunkManifest:
    @pytest.mark.xfail(
        reason="Tutorial data non coord dimensions are serialized with big endidan types and internally dropped"
    )
    def test_empty_chunks(self, empty_chunks_hdf5_file):
        store = obstore_local(file_url=empty_chunks_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=empty_chunks_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_empty_dataset(self, empty_dataset_hdf5_file):
        store = obstore_local(file_url=empty_dataset_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=empty_dataset_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_no_chunking(self, no_chunks_hdf5_file):
        store = obstore_local(file_url=no_chunks_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=no_chunks_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_file):
        store = obstore_local(file_url=chunked_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=chunked_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_file):
        store = obstore_local(file_url=chunked_roundtrip_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=chunked_roundtrip_hdf5_file, object_store=store
        )
        assert manifest_store._group.arrays["var2"].manifest.shape_chunk_grid == (2, 8)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_file):
        store = obstore_local(file_url=single_dimension_scale_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=single_dimension_scale_hdf5_file, object_store=store
        )
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("x",)

    def test_is_dimension_scale(self, is_scale_hdf5_file):
        store = obstore_local(file_url=is_scale_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=is_scale_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "data",
        )

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_file):
        store = obstore_local(file_url=multiple_dimension_scales_hdf5_file)
        parser = HDFParser()
        with pytest.raises(ValueError, match="dimension scales attached"):
            parser(file_url=multiple_dimension_scales_hdf5_file, object_store=store)

    def test_no_dimension_scales(self, no_chunks_hdf5_file):
        store = obstore_local(file_url=no_chunks_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=no_chunks_hdf5_file, object_store=store)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "phony_dim_0",
            "phony_dim_1",
        )


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetToManifestArray:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_file):
        store = obstore_local(file_url=chunked_dimensions_netcdf4_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=chunked_dimensions_netcdf4_file, object_store=store
        )
        assert manifest_store._group.arrays["data"].chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_file):
        store = obstore_local(file_url=single_dimension_scale_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=single_dimension_scale_hdf5_file, object_store=store
        )
        assert manifest_store._group.arrays["data"].chunks == (2,)

    def test_dataset_attributes(self, string_attributes_hdf5_file):
        store = obstore_local(file_url=string_attributes_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=string_attributes_hdf5_file, object_store=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.attributes["attribute_name"] == "attribute_name"

    def test_scalar_fill_value(self, scalar_fill_value_hdf5_file):
        store = obstore_local(file_url=scalar_fill_value_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=scalar_fill_value_hdf5_file, object_store=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.fill_value == 42

    def test_cf_fill_value(self, cf_fill_value_hdf5_file):
        f = h5py.File(cf_fill_value_hdf5_file)
        ds = f["data"]
        if ds.dtype.kind in "S":
            pytest.xfail("Investigate fixed-length binary encoding in Zarr v3")
        if ds.dtype.names:
            pytest.xfail("To fix, structured dtype fill value encoding for Zarr parser")
        store = obstore_local(file_url=cf_fill_value_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=cf_fill_value_hdf5_file, object_store=store)
        metadata = manifest_store._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes

    def test_cf_array_fill_value(self, cf_array_fill_value_hdf5_file):
        store = obstore_local(file_url=cf_array_fill_value_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=cf_array_fill_value_hdf5_file, object_store=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert not isinstance(metadata.attributes["_FillValue"], np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
class TestExtractAttributes:
    def test_root_attribute(self, root_attributes_hdf5_file):
        store = obstore_local(file_url=root_attributes_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(file_url=root_attributes_hdf5_file, object_store=store)
        assert (
            manifest_store._group.metadata.attributes["attribute_name"]
            == "attribute_name"
        )

    def test_multiple_attributes(self, string_attributes_hdf5_file):
        store = obstore_local(file_url=string_attributes_hdf5_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=string_attributes_hdf5_file, object_store=store
        )
        metadata = manifest_store._group.arrays["data"].metadata
        assert len(metadata.attributes.keys()) == 2


@requires_hdf5plugin
@requires_imagecodecs
class TestManifestGroupFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_file):
        store = obstore_local(file_url=chunked_dimensions_netcdf4_file)
        parser = HDFParser()
        manifest_store = parser(
            file_url=chunked_dimensions_netcdf4_file, object_store=store
        )
        assert len(manifest_store._group.arrays) == 3

    def test_nested_groups_are_ignored(self, nested_group_hdf5_file):
        store = obstore_local(file_url=nested_group_hdf5_file)
        parser = HDFParser(group="group")
        manifest_store = parser(file_url=nested_group_hdf5_file, object_store=store)
        assert len(manifest_store._group.arrays) == 1

    def test_drop_variables(self, multiple_datasets_hdf5_file):
        store = obstore_local(file_url=multiple_datasets_hdf5_file)
        parser = HDFParser(drop_variables=["data2"])
        manifest_store = parser(
            file_url=multiple_datasets_hdf5_file, object_store=store
        )
        assert "data2" not in manifest_store._group.arrays.keys()

    def test_dataset_in_group(self, group_hdf5_file):
        store = obstore_local(file_url=group_hdf5_file)
        parser = HDFParser(group="group")
        manifest_store = parser(file_url=group_hdf5_file, object_store=store)
        assert len(manifest_store._group.arrays) == 1

    def test_non_group_error(self, group_hdf5_file):
        store = obstore_local(file_url=group_hdf5_file)
        parser = HDFParser(group="group/data")
        with pytest.raises(ValueError):
            parser(file_url=group_hdf5_file, object_store=store)


@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualDataset:
    def test_coord_names(
        self,
        root_coordinates_hdf5_file,
    ):
        store = obstore_local(file_url=root_coordinates_hdf5_file)
        parser = HDFParser()
        with open_virtual_dataset(
            file_url=root_coordinates_hdf5_file,
            object_store=store,
            parser=parser,
        ) as vds:
            assert set(vds.coords) == {"lat", "lon"}

    @pytest.mark.xfail(reason="Requires Zarr v3 big endian dtype support")
    def test_big_endian(
        self,
        big_endian_dtype_hdf5_file,
    ):
        store = obstore_local(file_url=big_endian_dtype_hdf5_file)
        parser = HDFParser()
        with open_virtual_dataset(
            file_url=big_endian_dtype_hdf5_file,
            object_store=store,
            parser=parser,
        ) as vds:
            print(vds)


@requires_hdf5plugin
@requires_imagecodecs
@pytest.mark.parametrize("group", [None, "/", "subgroup", "subgroup/", "/subgroup/"])
def test_subgroup_variable_names(netcdf4_file_with_data_in_multiple_groups, group):
    # regression test for GH issue #364
    store = obstore_local(file_url=netcdf4_file_with_data_in_multiple_groups)
    parser = HDFParser(group=group)
    with open_virtual_dataset(
        file_url=netcdf4_file_with_data_in_multiple_groups,
        object_store=store,
        parser=parser,
    ) as vds:
        assert list(vds.dims) == ["dim_0"]
