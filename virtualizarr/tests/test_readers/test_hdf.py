import h5py
import pytest

from virtualizarr.readers.hdf import (_dataset_chunk_manifest, _dataset_dims,
                                      _dataset_to_variable, _extract_attrs)


class TestDatasetChunkManifest:
    def test_empty_chunks(self, empty_chunks_netcdf4_file):
        f = h5py.File(empty_chunks_netcdf4_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="chunked but contains no chunks"):
            _dataset_chunk_manifest(path=empty_chunks_netcdf4_file, dataset=ds)

    def test_empty_dataset(self, empty_dataset_netcdf4_file):
        f = h5py.File(empty_dataset_netcdf4_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="no space allocated in the file"):
            _dataset_chunk_manifest(path=empty_dataset_netcdf4_file, dataset=ds)

    def test_no_chunking(self, no_chunks_netcdf4_file):
        f = h5py.File(no_chunks_netcdf4_file)
        ds = f["data"]
        manifest = _dataset_chunk_manifest(path=no_chunks_netcdf4_file, dataset=ds)
        assert len(manifest.entries) == 1

    def test_chunked(self, chunked_netcdf4_file):
        f = h5py.File(chunked_netcdf4_file)
        ds = f["data"]
        manifest = _dataset_chunk_manifest(path=chunked_netcdf4_file, dataset=ds)
        assert len(manifest.entries) == 4


class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_netcdf4_file):
        f = h5py.File(single_dimension_scale_netcdf4_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims[0] == "x"

    def test_is_dimension_scale(self, is_scale_netcdf4_file):
        f = h5py.File(is_scale_netcdf4_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims[0] == "data"

    def test_multiple_dimension_scales(self, multiple_dimension_scales_netcdf4_file):
        f = h5py.File(multiple_dimension_scales_netcdf4_file)
        ds = f["data"]
        with pytest.raises(ValueError, match="dimension scales attached"):
            _dataset_dims(ds)

    def test_no_dimension_scales(self, no_chunks_netcdf4_file):
        f = h5py.File(no_chunks_netcdf4_file)
        ds = f["data"]
        dims = _dataset_dims(ds)
        assert dims == ["phony_dim_0", "phony_dim_1"]


class TestDatasetToVariable:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_file):
        f = h5py.File(chunked_dimensions_netcdf4_file)
        ds = f["data"]
        var = _dataset_to_variable(chunked_dimensions_netcdf4_file, ds)
        assert var.chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_netcdf4_file):
        f = h5py.File(single_dimension_scale_netcdf4_file)
        ds = f["data"]
        var = _dataset_to_variable(single_dimension_scale_netcdf4_file, ds)
        assert var.chunks == (2,)

    def test_dataset_attributes(self, string_attribute_netcdf4_file):
        f = h5py.File(string_attribute_netcdf4_file)
        ds = f["data"]
        var = _dataset_to_variable(string_attribute_netcdf4_file, ds)
        assert var.attrs["attribute_name"] == "attribute_name"


class TestExtractAttributes:
    def test_string_attribute(self, string_attribute_netcdf4_file):
        f = h5py.File(string_attribute_netcdf4_file)
        ds = f["data"]
        attrs = _extract_attrs(ds)
        assert attrs["attribute_name"] == "attribute_name"
