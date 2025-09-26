import functools
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray import Dataset, open_dataset
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import (
    requires_dask,
    requires_hdf5plugin,
    requires_imagecodecs,
    requires_lithops,
    requires_network,
    slow_test,
)
from virtualizarr.tests.utils import obstore_http, obstore_s3


def test_wrapping(array_v3_metadata):
    chunks = (5, 10)
    shape = (5, 20)
    dtype = np.dtype("int32")

    chunks_dict = {
        "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(
        metadata=array_v3_metadata(chunks=chunks, shape=shape), chunkmanifest=manifest
    )
    ds = xr.Dataset({"a": (["x", "y"], marr)})

    assert isinstance(ds["a"].data, ManifestArray)
    assert ds["a"].shape == shape
    assert ds["a"].dtype == dtype
    assert ds["a"].chunks == chunks


class TestEquals:
    # regression test for GH29 https://github.com/zarr-developers/VirtualiZarr/issues/29
    def test_equals(self, array_v3_metadata):
        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(metadata=array_v3_metadata(), chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        marr2 = ManifestArray(metadata=array_v3_metadata(), chunkmanifest=manifest1)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})
        assert ds1.equals(ds2)

        chunks_dict3 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest3 = ChunkManifest(entries=chunks_dict3)
        marr3 = ManifestArray(metadata=array_v3_metadata(), chunkmanifest=manifest3)
        ds3 = xr.Dataset({"a": (["x", "y"], marr3)})
        assert not ds1.equals(ds3)


# TODO refactor these tests by making some fixtures
class TestConcat:
    def test_concat_along_existing_dim(self, array_v3_metadata):
        # both manifest arrays in this example have the same metadata properties
        metadata = array_v3_metadata(chunks=(1, 10), shape=(1, 20))

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="x")["a"]
        assert result.indexes == {}

        assert result.shape == (2, 20)
        assert result.chunks == (1, 10)
        assert result.data.manifest.dict() == {
            "0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "1.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "1.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        metadata_copy = metadata.to_dict().copy()
        metadata_copy["shape"] = (2, 20)
        assert result.data.metadata.to_dict() == metadata_copy

    def test_concat_along_new_dim(self, array_v3_metadata):
        # this calls np.stack internally
        # both manifest arrays in this example have the same metadata properties

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        metadata = array_v3_metadata(chunks=(5, 10), shape=(5, 20))
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="z")["a"]
        assert result.indexes == {}

        # xarray.concat adds new dimensions along axis=0
        assert result.shape == (2, 5, 20)
        assert result.chunks == (1, 5, 10)
        assert result.data.manifest.dict() == {
            "0.0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "1.0.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "1.0.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        metadata_copy = metadata.to_dict().copy()
        metadata_copy["shape"] = (2, 5, 20)
        metadata_copy["chunk_grid"]["configuration"]["chunk_shape"] = (1, 5, 10)
        assert result.data.metadata.to_dict() == metadata_copy

    def test_concat_dim_coords_along_existing_dim(self, array_v3_metadata):
        # Tests that dimension coordinates don't automatically get new indexes on concat
        # See https://github.com/pydata/xarray/issues/8871

        # both manifest arrays in this example have the same metadata properties

        chunks_dict1 = {
            "0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        metadata = array_v3_metadata(chunks=(10,), shape=(20,))
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest1)
        coords = xr.Coordinates({"t": (["t"], marr1)}, indexes={})
        ds1 = xr.Dataset(coords=coords)

        chunks_dict2 = {
            "0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(metadata=metadata, chunkmanifest=manifest2)
        coords = xr.Coordinates({"t": (["t"], marr2)}, indexes={})
        ds2 = xr.Dataset(coords=coords)

        result = xr.concat([ds1, ds2], dim="t")["t"]
        assert result.indexes == {}

        assert result.shape == (40,)
        assert result.chunks == (10,)
        assert result.data.manifest.dict() == {
            "0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "2": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "3": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        metadata_copy = metadata.to_dict().copy()
        metadata_copy["shape"] = (40,)
        metadata_copy["chunk_grid"]["configuration"]["chunk_shape"] = (10,)
        assert result.data.metadata.to_dict() == metadata_copy


@requires_hdf5plugin
@requires_imagecodecs
class TestCombine:
    def test_combine_by_coords(
        self, netcdf4_files_factory: Callable[[], tuple[str, str]], local_registry
    ):
        filepath1, filepath2 = netcdf4_files_factory()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=filepath1,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds1,
            open_virtual_dataset(
                url=filepath2,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds2,
        ):
            combined_vds = xr.combine_by_coords(
                [vds2, vds1],
            )

            assert (
                combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing
            )

    def test_2d_combine_by_coords(
        self,
        netcdf4_files_factory_2d: Callable[[], tuple[str, str, str, str]],
        local_registry,
    ):
        filepath1, filepath2, filepath3, filepath4 = netcdf4_files_factory_2d()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=filepath1,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds1,
            open_virtual_dataset(
                url=filepath2,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds2,
            open_virtual_dataset(
                url=filepath3,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds3,
            open_virtual_dataset(
                url=filepath4,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds4,
        ):
            combined_vds = xr.combine_by_coords(
                [vds2, vds1, vds4, vds3],
                coords="minimal",
                compat="override",
                join="override",
            )
            assert combined_vds.sizes == {"lat": 20, "time": 2920, "lon": 53}
            assert (
                combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing
            )
            assert (
                combined_vds.xindexes["lat"].to_pandas_index().is_monotonic_decreasing
            )

    def test_2d_combine_nested(
        self,
        netcdf4_files_factory_2d: Callable[[], tuple[str, str, str, str]],
        local_registry,
    ):
        filepath1, filepath2, filepath3, filepath4 = netcdf4_files_factory_2d()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=filepath1,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds1,
            open_virtual_dataset(
                url=filepath2,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds2,
            open_virtual_dataset(
                url=filepath3,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds3,
            open_virtual_dataset(
                url=filepath4,
                registry=local_registry,
                parser=parser,
                loadable_variables=["time", "lat", "lon"],
            ) as vds4,
        ):
            combined_vds = xr.combine_nested(
                [
                    [vds1, vds3],
                    [vds2, vds4],
                ],
                concat_dim=["time", "lat"],
                coords="minimal",
                compat="override",
                join="override",
            )
            assert combined_vds.sizes == {"lat": 20, "time": 2920, "lon": 53}
            assert (
                combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing
            )
            assert (
                combined_vds.xindexes["lat"].to_pandas_index().is_monotonic_decreasing
            )

    @pytest.mark.xfail(reason="Not yet implemented, see issue #18")
    def test_combine_by_coords_keeping_manifestarrays(
        self, netcdf4_files_factory: Callable[[], tuple[str, str]], local_registry
    ):
        filepath1, filepath2 = netcdf4_files_factory()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=filepath1, registry=local_registry, parser=parser
            ) as vds1,
            open_virtual_dataset(
                url=filepath2, registry=local_registry, parser=parser
            ) as vds2,
        ):
            combined_vds = xr.combine_by_coords([vds2, vds1])

            assert isinstance(combined_vds["time"].data, ManifestArray)
            assert isinstance(combined_vds["lat"].data, ManifestArray)
            assert isinstance(combined_vds["lon"].data, ManifestArray)


class TestRenamePaths:
    def test_old_accessor(self, netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            with pytest.warns(DeprecationWarning):
                renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
                assert (
                    renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                    == "s3://bucket/air.nc"
                )

    def test_rename_to_str(self, netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            renamed_vds = vds.vz.rename_paths("s3://bucket/air.nc")
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )

    def test_rename_using_function(self, netcdf4_file, local_registry):
        def local_to_s3_url(old_local_path: str) -> str:
            from pathlib import Path

            new_s3_bucket_url = "s3://bucket/"
            filename = Path(old_local_path).name
            return str(new_s3_bucket_url + filename)

        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            renamed_vds = vds.vz.rename_paths(local_to_s3_url)
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )

    def test_invalid_type(self, netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file, registry=local_registry, parser=parser
        ) as vds:
            with pytest.raises(TypeError):
                vds.vz.rename_paths(["file1.nc", "file2.nc"])

    @requires_hdf5plugin
    @requires_imagecodecs
    def test_mixture_of_manifestarrays_and_numpy_arrays(
        self, netcdf4_file, local_registry
    ):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
            loadable_variables=["lat", "lon"],
        ) as vds:
            renamed_vds = vds.vz.rename_paths("s3://bucket/air.nc")
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )
            assert isinstance(renamed_vds["lat"].data, np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
def test_nbytes(simple_netcdf4, local_registry):
    parser = HDFParser()
    with open_virtual_dataset(
        url=simple_netcdf4,
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert vds.vz.nbytes == 32
        assert vds.nbytes == 48

    with open_virtual_dataset(
        url=simple_netcdf4,
        registry=local_registry,
        parser=parser,
        loadable_variables=["foo"],
    ) as vds:
        assert vds.vz.nbytes == 48

    with open_dataset(simple_netcdf4) as ds:
        assert ds.vz.nbytes == ds.nbytes


class TestOpenVirtualDatasetIndexes:
    @requires_hdf5plugin
    @requires_imagecodecs
    def test_create_default_indexes_for_loadable_variables(
        self, netcdf4_file, local_registry
    ):
        loadable_variables = ["time", "lat"]

        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=netcdf4_file,
                registry=local_registry,
                parser=parser,
                loadable_variables=loadable_variables,
            ) as vds,
            open_dataset(netcdf4_file, decode_times=True) as ds,
        ):
            # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
            assert index_mappings_equal(vds.xindexes, ds[loadable_variables].xindexes)


def index_mappings_equal(indexes1: Mapping[str, Index], indexes2: Mapping[str, Index]):
    # Check if the mappings have the same keys
    if set(indexes1.keys()) != set(indexes2.keys()):
        return False

    # Check if the values for each key are identical
    for key in indexes1.keys():
        index1 = indexes1[key]
        index2 = indexes2[key]

        if not index1.equals(index2):
            return False

    return True


@requires_hdf5plugin
@requires_imagecodecs
def test_cftime_index(tmp_path: Path, local_registry):
    """Ensure a virtual dataset contains the same indexes as an Xarray dataset"""
    # Note: Test was created to debug: https://github.com/zarr-developers/VirtualiZarr/issues/168
    filepath = str(tmp_path / "tmp.nc")
    ds = xr.Dataset(
        data_vars={
            "tasmax": (["time", "lat", "lon"], np.random.rand(2, 18, 36)),
        },
        coords={
            "time": np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]"),
            "lat": np.arange(-90, 90, 10),
            "lon": np.arange(-180, 180, 10),
        },
        attrs={"attr1_key": "attr1_val"},
    )
    ds.to_netcdf(filepath)

    parser = HDFParser()
    with open_virtual_dataset(
        url=filepath,
        registry=local_registry,
        parser=parser,
        loadable_variables=["time", "lat", "lon"],
    ) as vds:
        # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
        assert index_mappings_equal(vds.xindexes, ds.xindexes)
        assert list(ds.coords) == list(vds.coords)
        assert vds.dims == ds.dims
        assert vds.attrs == ds.attrs


class TestOpenVirtualDatasetAttrs:
    def test_drop_array_dimensions(self, netcdf4_file, local_registry):
        parser = HDFParser()
        # regression test for GH issue #150
        vds = open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        )
        assert "_ARRAY_DIMENSIONS" not in vds["air"].attrs

    def test_coordinate_variable_attrs_preserved(self, netcdf4_file, local_registry):
        # regression test for GH issue #155
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert vds["lat"].attrs == {
                "standard_name": "latitude",
                "long_name": "Latitude",
                "units": "degrees_north",
                "axis": "Y",
            }


class TestDetermineCoords:
    def test_infer_one_dimensional_coords(self, netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert set(vds.coords) == {"time", "lat", "lon"}

    def test_var_attr_coords(self, netcdf4_file_with_2d_coords, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file_with_2d_coords,
            registry=local_registry,
            parser=parser,
        ) as vds:
            expected_dimension_coords = ["ocean_time", "s_rho"]
            expected_2d_coords = ["lon_rho", "lat_rho", "h"]
            expected_1d_non_dimension_coords = ["Cs_r"]
            expected_scalar_coords = ["hc", "Vtransform"]
            expected_coords = (
                expected_dimension_coords
                + expected_2d_coords
                + expected_1d_non_dimension_coords
                + expected_scalar_coords
            )
            assert set(vds.coords) == set(expected_coords)


@requires_network
class TestReadRemote:
    @slow_test
    @pytest.mark.parametrize(
        "indexes",
        [
            None,
            pytest.param({}, marks=pytest.mark.xfail(reason="not implemented")),
        ],
        ids=["None index", "empty dict index"],
    )
    def test_anon_read_s3(self, indexes):
        """Parameterized tests for empty vs supplied indexes and filetypes."""
        # TODO: Switch away from this s3 url after minIO is implemented.
        filepath = "s3://carbonplan-share/virtualizarr/local.nc"
        object_store = obstore_s3(url=filepath, region="us-west-2")
        registry = ObjectStoreRegistry()
        registry.register(filepath, object_store)
        parser = HDFParser()
        with open_virtual_dataset(
            url=filepath,
            registry=registry,
            indexes=indexes,
            parser=parser,
        ) as vds:
            assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}

            assert isinstance(vds["air"].data, ManifestArray)
            for name in ["time", "lat", "lon"]:
                assert isinstance(vds[name].data, np.ndarray)

    @slow_test
    def test_virtualizarr_vs_local_nisar(self):
        # Open group directly from locally cached file with xarray
        url = "https://nisar.asf.earthdatacloud.nasa.gov/NISAR-SAMPLE-DATA/GCOV/ALOS1_Rosamond_20081012/NISAR_L2_PR_GCOV_001_005_A_219_4020_SHNA_A_20081012T060910_20081012T060926_P01101_F_N_J_001.h5"
        hdf_group = "science/LSAR/GCOV/grids/frequencyA"
        store = obstore_http(url=url)
        registry = ObjectStoreRegistry()
        registry.register(url, store)
        drop_variables = ["listOfCovarianceTerms", "listOfPolarizations"]
        parser = HDFParser(group=hdf_group, drop_variables=drop_variables)
        with (
            xr.open_dataset(
                url,
                engine="h5netcdf",
                group=hdf_group,
                drop_variables=drop_variables,
                phony_dims="access",
            ) as dsXR,
            # save group reference file via virtualizarr, then open with engine="kerchunk"
            open_virtual_dataset(
                url=url,
                registry=registry,
                parser=parser,
            ) as vds,
        ):
            tmpref = "/tmp/cmip6.json"
            vds.vz.to_kerchunk(tmpref, format="json")

            with xr.open_dataset(tmpref, engine="kerchunk") as dsV:
                # xrt.assert_identical(dsXR, dsV) #Attribute order changes
                xrt.assert_equal(dsXR, dsV)


class TestOpenVirtualDatasetHDFGroup:
    def test_open_empty_group(self, empty_netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=empty_netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert isinstance(vds, xr.Dataset)
            expected = Dataset()
            xrt.assert_identical(vds, expected)

    def test_open_subgroup(
        self, netcdf4_file_with_data_in_multiple_groups, local_registry
    ):
        parser = HDFParser(group="subgroup")
        with open_virtual_dataset(
            url=netcdf4_file_with_data_in_multiple_groups,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert list(vds.variables) == ["bar"]
            assert isinstance(vds["bar"].data, ManifestArray)
            assert vds["bar"].shape == (2,)

    @pytest.mark.parametrize("group", ["", None])
    def test_open_root_group(
        self, netcdf4_file_with_data_in_multiple_groups, group, local_registry
    ):
        parser = HDFParser(group=group)
        with open_virtual_dataset(
            url=netcdf4_file_with_data_in_multiple_groups,
            registry=local_registry,
            parser=parser,
        ) as vds:
            assert list(vds.variables) == ["foo"]
            assert isinstance(vds["foo"].data, ManifestArray)
            assert vds["foo"].shape == (3,)


@requires_hdf5plugin
@requires_imagecodecs
class TestLoadVirtualDataset:
    @pytest.mark.parametrize(
        "loadable_variables, expected_loadable_variables",
        [
            ([], []),
            (["time"], ["time"]),
            (["air", "time"], ["air", "time"]),
            (None, ["lat", "lon", "time"]),
        ],
    )
    def test_loadable_variables(
        self,
        netcdf4_file,
        loadable_variables,
        expected_loadable_variables,
        local_registry,
    ):
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=netcdf4_file,
                registry=local_registry,
                loadable_variables=loadable_variables,
                parser=parser,
            ) as vds,
            xr.open_dataset(netcdf4_file, decode_times=True) as ds,
        ):
            assert set(vds.variables) == set(ds.variables)
            assert set(vds.coords) == set(ds.coords)

            virtual_variables = {
                name: var
                for name, var in vds.variables.items()
                if isinstance(var.data, ManifestArray)
            }
            actual_loadable_variables = {
                name: var
                for name, var in vds.variables.items()
                if not isinstance(var.data, ManifestArray)
            }

            assert set(actual_loadable_variables) == set(expected_loadable_variables)

            for var in virtual_variables.values():
                assert isinstance(var.data, ManifestArray)

            for name, var in ds.variables.items():
                if name in actual_loadable_variables:
                    xrt.assert_identical(vds.variables[name], ds.variables[name])

    def test_group_kwarg_not_a_group(self, hdf5_groups_file, local_registry):
        parser = HDFParser(group="doesnt_exist")
        with pytest.raises(ValueError, match="not an HDF Group"):
            with open_virtual_dataset(
                url=hdf5_groups_file,
                registry=local_registry,
                parser=parser,
            ):
                pass

    def test_group_kwarg(self, hdf5_groups_file, local_registry):
        parser = HDFParser(group="test/group")
        vars_to_load = ["air", "time"]
        with (
            open_virtual_dataset(
                url=hdf5_groups_file,
                registry=local_registry,
                loadable_variables=vars_to_load,
                parser=parser,
            ) as vds,
            xr.open_dataset(hdf5_groups_file, group="test/group") as full_ds,
        ):
            for name in full_ds.variables:
                if name in vars_to_load:
                    xrt.assert_identical(vds.variables[name], full_ds.variables[name])

    def test_open_dataset_with_empty(self, hdf5_empty, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=hdf5_empty, registry=local_registry, parser=parser
        ) as vds:
            assert vds.empty.dims == ()
            assert vds.empty.attrs == {"empty": "true"}

    def test_open_dataset_with_scalar(self, hdf5_scalar, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=f"file://{hdf5_scalar}", registry=local_registry, parser=parser
        ) as vds:
            assert vds.scalar.dims == ()
            assert vds.scalar.attrs == {"scalar": "true"}
            assert isinstance(vds.scalar.data, ManifestArray)
        ms = parser(registry=local_registry, url=f"file://{hdf5_scalar}")
        with (
            xr.open_dataset(hdf5_scalar, engine="h5netcdf") as expected,
            xr.open_zarr(ms, consolidated=False, zarr_format=3) as observed,
        ):
            xr.testing.assert_allclose(expected, observed)


preprocess_func = functools.partial(
    xr.Dataset.rename_vars,
    air="nair",
)


@requires_hdf5plugin
@requires_imagecodecs
class TestOpenVirtualMFDataset:
    @pytest.mark.parametrize("invalid_parallel_kwarg", ["ray", Dataset])
    def test_invalid_parallel_kwarg(
        self, netcdf4_files_factory, invalid_parallel_kwarg, local_registry
    ):
        filepath1, filepath2 = netcdf4_files_factory()
        parser = HDFParser()
        with pytest.raises(ValueError, match="Unrecognized argument"):
            open_virtual_mfdataset(
                [filepath1, filepath2],
                registry=local_registry,
                parser=parser,
                combine="nested",
                concat_dim="time",
                parallel=invalid_parallel_kwarg,
            )

    @pytest.mark.parametrize(
        "parallel",
        [
            False,
            ThreadPoolExecutor,
            pytest.param("dask", marks=requires_dask),
            pytest.param("lithops", marks=requires_lithops),
        ],
    )
    @pytest.mark.parametrize(
        "preprocess",
        [
            None,
            preprocess_func,
        ],
    )
    def test_parallel_open(
        self, netcdf4_files_factory, parallel, preprocess, local_registry
    ):
        if parallel == "lithops":
            pytest.xfail(
                "TODO - investigate intermittent test failures with lithops executor"
            )
        filepath1, filepath2 = netcdf4_files_factory()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=filepath1, registry=local_registry, parser=parser
            ) as vds1,
            open_virtual_dataset(
                url=filepath2,
                registry=local_registry,
                parser=parser,
            ) as vds2,
        ):
            expected_vds = xr.concat([vds1, vds2], dim="time")
            if preprocess:
                expected_vds = preprocess_func(expected_vds)

            # test combine nested, which doesn't use in-memory indexes
            combined_vds = open_virtual_mfdataset(
                [filepath1, filepath2],
                registry=local_registry,
                parser=parser,
                combine="nested",
                concat_dim="time",
                parallel=parallel,
                preprocess=preprocess,
            )
            xrt.assert_identical(combined_vds, expected_vds)

            # test combine by coords using in-memory indexes
            combined_vds = open_virtual_mfdataset(
                [filepath1, filepath2],
                registry=local_registry,
                parser=parser,
                combine="by_coords",
                parallel=parallel,
                preprocess=preprocess,
            )
            xrt.assert_identical(combined_vds, expected_vds)

            # test combine by coords again using in-memory indexes but for a glob
            file_glob = Path(filepath1).parent.glob("air*.nc")
            combined_vds = open_virtual_mfdataset(
                file_glob,
                registry=local_registry,
                parser=parser,
                combine="by_coords",
                parallel=parallel,
                preprocess=preprocess,
            )
            xrt.assert_identical(combined_vds, expected_vds)


def test_drop_variables(netcdf4_file, local_registry):
    parser = HDFParser()
    with open_virtual_dataset(
        url=netcdf4_file,
        registry=local_registry,
        parser=parser,
        drop_variables=["air"],
    ) as vds:
        assert "air" not in vds.variables


def test_concat_zero_dimensional_var(manifest_array):
    # regression test for https://github.com/zarr-developers/VirtualiZarr/pull/641
    marr = manifest_array(shape=(), chunks=())
    vds1 = xr.Dataset({"a": marr})
    vds2 = xr.Dataset({"a": marr})
    result = xr.concat([vds1, vds2], dim="time", coords="minimal", compat="override")
    assert result["a"].sizes == {"time": 2}
