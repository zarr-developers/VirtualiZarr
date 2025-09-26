import h5py  # type: ignore
import numpy as np
import pytest
import xarray as xr
from obstore.store import from_url

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)
from virtualizarr.tests.utils import manifest_store_from_hdf_url


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetChunkManifest:
    @pytest.mark.xfail(
        reason="Tutorial data non coord dimensions are serialized with big endidan types and internally dropped"
    )
    def test_empty_chunks(self, empty_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(empty_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_empty_dataset(self, empty_dataset_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(empty_dataset_hdf5_url)
        assert manifest_store._group.arrays["data"].shape == (0,)

    def test_no_chunking(self, no_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(no_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (1, 1)

    def test_chunked(self, chunked_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(chunked_hdf5_url)
        assert manifest_store._group.arrays["data"].manifest.shape_chunk_grid == (2, 2)

    def test_chunked_roundtrip(self, chunked_roundtrip_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(chunked_roundtrip_hdf5_url)
        assert manifest_store._group.arrays["var2"].manifest.shape_chunk_grid == (2, 8)


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetDims:
    def test_single_dimension_scale(self, single_dimension_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(single_dimension_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == ("x",)

    def test_is_dimension_scale(self, is_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(is_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "data",
        )

    def test_multiple_dimension_scales(self, multiple_dimension_scales_hdf5_url):
        with pytest.raises(ValueError, match="dimension scales attached"):
            manifest_store_from_hdf_url(multiple_dimension_scales_hdf5_url)

    def test_no_dimension_scales(self, no_chunks_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(no_chunks_hdf5_url)
        assert manifest_store._group.arrays["data"].metadata.dimension_names == (
            "phony_dim_0",
            "phony_dim_1",
        )


@requires_hdf5plugin
@requires_imagecodecs
class TestDatasetToManifestArray:
    def test_chunked_dataset(self, chunked_dimensions_netcdf4_url):
        manifest_store = manifest_store_from_hdf_url(chunked_dimensions_netcdf4_url)
        assert manifest_store._group.arrays["data"].chunks == (50, 50)

    def test_not_chunked_dataset(self, single_dimension_scale_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(single_dimension_scale_hdf5_url)
        assert manifest_store._group.arrays["data"].chunks == (2,)

    def test_dataset_attributes(self, string_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(string_attributes_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.attributes["attribute_name"] == "attribute_name"

    def test_scalar_fill_value(self, scalar_fill_value_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(scalar_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert metadata.fill_value == 42

    def test_cf_fill_value(self, cf_fill_value_hdf5_file):
        cf_fill_value_hdf5_url = f"file://{cf_fill_value_hdf5_file}"
        f = h5py.File(cf_fill_value_hdf5_file)
        ds = f["data"]
        if ds.dtype.kind in "S":
            pytest.xfail("Investigate fixed-length binary encoding in Zarr v3")
        if ds.dtype.names:
            pytest.xfail("To fix, structured dtype fill value encoding for Zarr parser")
        manifest_store = manifest_store_from_hdf_url(cf_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert "_FillValue" in metadata.attributes

    def test_cf_array_fill_value(self, cf_array_fill_value_hdf5_file):
        cf_array_fill_value_hdf5_url = f"file://{cf_array_fill_value_hdf5_file}"
        manifest_store = manifest_store_from_hdf_url(cf_array_fill_value_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert not isinstance(metadata.attributes["_FillValue"], np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
class TestExtractAttributes:
    def test_root_attribute(self, root_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(root_attributes_hdf5_url)
        assert (
            manifest_store._group.metadata.attributes["attribute_name"]
            == "attribute_name"
        )

    def test_multiple_attributes(self, string_attributes_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(string_attributes_hdf5_url)
        metadata = manifest_store._group.arrays["data"].metadata
        assert len(metadata.attributes.keys()) == 2


@requires_hdf5plugin
@requires_imagecodecs
class TestManifestGroupFromHDF:
    def test_variable_with_dimensions(self, chunked_dimensions_netcdf4_url):
        manifest_store = manifest_store_from_hdf_url(chunked_dimensions_netcdf4_url)
        assert len(manifest_store._group.arrays) == 3

    def test_nested_groups_are_ignored(self, nested_group_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(
            nested_group_hdf5_url, group="group"
        )
        assert len(manifest_store._group.arrays) == 1

    def test_drop_variables(self, multiple_datasets_hdf5_url, local_registry):
        parser = HDFParser(drop_variables=["data2"])
        manifest_store = parser(url=multiple_datasets_hdf5_url, registry=local_registry)
        assert "data2" not in manifest_store._group.arrays.keys()

    def test_dataset_in_group(self, group_hdf5_url):
        manifest_store = manifest_store_from_hdf_url(group_hdf5_url, group="group")
        assert len(manifest_store._group.arrays) == 1

    def test_non_group_error(self, group_hdf5_url):
        with pytest.raises(ValueError):
            manifest_store_from_hdf_url(group_hdf5_url, group="group/data")


@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualDataset:
    def test_coord_names(self, root_coordinates_hdf5_file, local_registry):
        root_coordinates_hdf5_url = f"file://{root_coordinates_hdf5_file}"
        parser = HDFParser()
        with open_virtual_dataset(
            url=root_coordinates_hdf5_url,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert set(vds.coords) == {"lat", "lon"}

    def test_big_endian(self, big_endian_dtype_hdf5_file, local_registry):
        big_endian_dtype_hdf5_url = f"file://{big_endian_dtype_hdf5_file}"
        parser = HDFParser()
        with (
            parser(
                url=big_endian_dtype_hdf5_url, registry=local_registry
            ) as manifest_store,
            xr.open_dataset(big_endian_dtype_hdf5_file) as expected,
        ):
            observed = xr.open_dataset(
                manifest_store, engine="zarr", consolidated=False, zarr_format=3
            )
            assert isinstance(observed, xr.Dataset)
            xr.testing.assert_identical(observed.load(), expected.load())


@requires_hdf5plugin
@requires_imagecodecs
@pytest.mark.parametrize("group", [None, "/", "subgroup", "subgroup/", "/subgroup/"])
def test_subgroup_variable_names(
    netcdf4_file_with_data_in_multiple_groups, group, local_registry
):
    # regression test for GH issue #364
    netcdf4_url_with_data_in_multiple_groups = (
        f"file://{netcdf4_file_with_data_in_multiple_groups}"
    )
    parser = HDFParser(group=group)
    with open_virtual_dataset(
        url=netcdf4_url_with_data_in_multiple_groups,
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert list(vds.dims) == ["dim_0"]


# @requires_network
def test_netcdf_over_https():
    url = "https://www.earthbyte.org/webdav/gmt_mirror/gmt/data/cache/topo_32.nc"
    store = from_url(url)
    registry = ObjectStoreRegistry({url: store})
    parser = HDFParser()
    with (
        parser(url=url, registry=registry) as ms,
        xr.open_zarr(ms, zarr_format=3, consolidated=False).load() as ds,
    ):
        np.testing.assert_allclose(ds["z"].min().to_numpy(), -6)
        np.testing.assert_allclose(ds["z"].max().to_numpy(), 817)
