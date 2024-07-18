import h5py
import pytest

from virtualizarr.readers.hdf import (
    _dataset_chunk_manifest,
    _dataset_dims,
    _dataset_to_variable,
    _extract_attrs,
    virtual_vars_from_hdf,
)


class TestDatasetChunkManifest:
    def test_empty_chunks(self, empty_chunks_hdf5_file):
        f = h5py.File(empty_chunks_hdf5_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="chunked but contains no chunks"):
            _dataset_chunk_manifest(path=empty_chunks_hdf5_file, dataset=ds)

    @pytest.mark.skip("Need to differentiate non coordinate dimensions from empty")
    def test_empty_dataset(self, empty_dataset_hdf5_file):
        f = h5py.File(empty_dataset_hdf5_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="no space allocated in the file"):
            _dataset_chunk_manifest(path=empty_dataset_hdf5_file, dataset=ds)

    def test_no_chunking(self, no_chunks_hdf5_file):
        f = h5py.File(no_chunks_hdf5_file)
        ds = f["data"]
        manifest = _dataset_chunk_manifest(path=no_chunks_hdf5_file, dataset=ds)
        assert manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_file):
        f = h5py.File(chunked_hdf5_file)
        ds = f["data"]
        manifest = _dataset_chunk_manifest(path=chunked_hdf5_file, dataset=ds)
        assert manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_file):
        f = h5py.File(chunked_roundtrip_hdf5_file)
        ds = f["var2"]
        manifest = _dataset_chunk_manifest(path=chunked_roundtrip_hdf5_file, dataset=ds)
        assert manifest.shape_chunk_grid == (2, 8)


class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_file):
        f = h5py.File(single_dimension_scale_hdf5_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims[0] == "x"

    def test_is_dimension_scale(self, is_scale_hdf5_file):
        f = h5py.File(is_scale_hdf5_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims[0] == "data"

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_file):
        f = h5py.File(multiple_dimension_scales_hdf5_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="dimension scales attached"):
            _dataset_dims(ds)

    def test_no_dimension_scales(self, no_chunks_hdf5_file):
        f = h5py.File(no_chunks_hdf5_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims == ["phony_dim_0", "phony_dim_1"]


class TestDatasetToVariable:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_file):
        f = h5py.File(chunked_dimensions_netcdf4_file)
        ds = f["data"]
        var = _dataset_to_variable(chunked_dimensions_netcdf4_file, ds)
        assert var.chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_file):
        f = h5py.File(single_dimension_scale_hdf5_file)
        ds = f["data"]
        var = _dataset_to_variable(single_dimension_scale_hdf5_file, ds)
        assert var.chunks == (2,)

    def test_dataset_attributes(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        var = _dataset_to_variable(string_attributes_hdf5_file, ds)
        assert var.attrs["attribute_name"] == "attribute_name"


class TestExtractAttributes:
    def test_string_attribute(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        attrs = _extract_attrs(ds)
        assert attrs["attribute_name"] == "attribute_name"

    def test_root_attribute(self, root_attributes_hdf5_file):
        f = h5py.File(root_attributes_hdf5_file)
        attrs = _extract_attrs(f)
        assert attrs["attribute_name"] == "attribute_name"

    def test_multiple_attributes(self, string_attributes_hdf5_file):
        f = h5py.File(string_attributes_hdf5_file)
        ds = f["data"]
        attrs = _extract_attrs(ds)
        assert len(attrs.keys()) == 2


class TestVirtualVarsFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_file):
        variables = virtual_vars_from_hdf(chunked_dimensions_netcdf4_file)
        assert len(variables) == 3

    def test_groups_not_implemented(self, group_hdf5_file):
        with pytest.raises(NotImplementedError):
            virtual_vars_from_hdf(group_hdf5_file)

    def test_drop_variables(self, multiple_datasets_hdf5_file):
        variables = virtual_vars_from_hdf(multiple_datasets_hdf5_file, ["data2"])
        assert "data2" not in variables.keys()
