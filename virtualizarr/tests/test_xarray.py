from collections.abc import Mapping

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from xarray.core.indexes import Index

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import network, requires_s3fs
from virtualizarr.zarr import ZArray


def test_wrapping():
    chunks = (5, 10)
    shape = (5, 20)
    dtype = np.dtype("int32")
    zarray = ZArray(
        chunks=chunks,
        compressor="zlib",
        dtype=dtype,
        fill_value=0.0,
        filters=None,
        order="C",
        shape=shape,
        zarr_format=2,
    )

    chunks_dict = {
        "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(zarray=zarray, chunkmanifest=manifest)
    ds = xr.Dataset({"a": (["x", "y"], marr)})

    assert isinstance(ds["a"].data, ManifestArray)
    assert ds["a"].shape == shape
    assert ds["a"].dtype == dtype
    assert ds["a"].chunks == chunks


class TestEquals:
    # regression test for GH29 https://github.com/TomNicholas/VirtualiZarr/issues/29
    def test_equals(self):
        chunks = (5, 10)
        shape = (5, 20)
        zarray = ZArray(
            chunks=chunks,
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})
        assert ds1.equals(ds2)

        chunks_dict3 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest3 = ChunkManifest(entries=chunks_dict3)
        marr3 = ManifestArray(zarray=zarray, chunkmanifest=manifest3)
        ds3 = xr.Dataset({"a": (["x", "y"], marr3)})
        assert not ds1.equals(ds3)


# TODO refactor these tests by making some fixtures
class TestConcat:
    def test_concat_along_existing_dim(self):
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(1, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="x")["a"]
        assert result.indexes == {}

        assert result.shape == (2, 20)
        assert result.chunks == (1, 10)
        assert result.data.manifest.dict() == {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "1.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format

    def test_concat_along_new_dim(self):
        # this calls np.stack internally
        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(5, 10),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})

        result = xr.concat([ds1, ds2], dim="z")["a"]
        assert result.indexes == {}

        # xarray.concat adds new dimensions along axis=0
        assert result.shape == (2, 5, 20)
        assert result.chunks == (1, 5, 10)
        assert result.data.manifest.dict() == {
            "0.0.0": {"path": "foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "foo.nc", "offset": 200, "length": 100},
            "1.0.0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1.0.1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format

    def test_concat_dim_coords_along_existing_dim(self):
        # Tests that dimension coordinates don't automatically get new indexes on concat
        # See https://github.com/pydata/xarray/issues/8871

        # both manifest arrays in this example have the same zarray properties
        zarray = ZArray(
            chunks=(10,),
            compressor="zlib",
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(20,),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0": {"path": "foo.nc", "offset": 100, "length": 100},
            "1": {"path": "foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        coords = xr.Coordinates({"t": (["t"], marr1)}, indexes={})
        ds1 = xr.Dataset(coords=coords)

        chunks_dict2 = {
            "0": {"path": "foo.nc", "offset": 300, "length": 100},
            "1": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
        coords = xr.Coordinates({"t": (["t"], marr2)}, indexes={})
        ds2 = xr.Dataset(coords=coords)

        result = xr.concat([ds1, ds2], dim="t")["t"]
        assert result.indexes == {}

        assert result.shape == (40,)
        assert result.chunks == (10,)
        assert result.data.manifest.dict() == {
            "0": {"path": "foo.nc", "offset": 100, "length": 100},
            "1": {"path": "foo.nc", "offset": 200, "length": 100},
            "2": {"path": "foo.nc", "offset": 300, "length": 100},
            "3": {"path": "foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format


class TestOpenVirtualDatasetIndexes:
    def test_no_indexes(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes={})
        assert vds.indexes == {}

    def test_create_default_indexes(self, netcdf4_file):
        vds = open_virtual_dataset(netcdf4_file, indexes=None)
        ds = xr.open_dataset(netcdf4_file)

        # TODO use xr.testing.assert_identical(vds.indexes, ds.indexes) instead once class supported by assertion comparison, see https://github.com/pydata/xarray/issues/5812
        assert index_mappings_equal(vds.xindexes, ds.xindexes)


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


class TestCombineUsingIndexes:
    def test_combine_by_coords(self, netcdf4_files):
        filepath1, filepath2 = netcdf4_files

        vds1 = open_virtual_dataset(filepath1)
        vds2 = open_virtual_dataset(filepath2)

        combined_vds = xr.combine_by_coords(
            [vds2, vds1],
        )

        assert combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing


@network
@requires_s3fs
class TestReadFromS3:
    @pytest.mark.parametrize(
        "filetype", ["netcdf4", None], ids=["netcdf4 filetype", "None filetype"]
    )
    @pytest.mark.parametrize(
        "indexes", [None, {}], ids=["None index", "empty dict index"]
    )
    def test_anon_read_s3(self, filetype, indexes):
        """Parameterized tests for empty vs supplied indexes and filetypes."""
        # TODO: Switch away from this s3 url after minIO is implemented.
        fpath = "s3://carbonplan-share/virtualizarr/local.nc"
        vds = open_virtual_dataset(fpath, filetype=filetype, indexes=indexes)

        assert vds.dims == {"time": 2920, "lat": 25, "lon": 53}
        for var in vds.variables:
            assert isinstance(vds[var].data, ManifestArray), var


class TestLoadVirtualDataset:
    def test_loadable_variables(self, netcdf4_file):
        vars_to_load = ["air", "time"]
        vds = open_virtual_dataset(netcdf4_file, loadable_variables=vars_to_load)

        for name in vds.variables:
            if name in vars_to_load:
                assert isinstance(vds[name].data, np.ndarray), name
            else:
                assert isinstance(vds[name].data, ManifestArray), name

        full_ds = xr.open_dataset(netcdf4_file)

        for name in full_ds.variables:
            if name in vars_to_load:
                xrt.assert_identical(vds.variables[name], full_ds.variables[name])
