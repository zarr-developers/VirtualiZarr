from typing import Callable

import numpy as np
import pytest
import xarray as xr

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.readers.hdf import HDFVirtualBackend
from virtualizarr.tests import requires_kerchunk
from virtualizarr.zarr import ZArray


def test_wrapping():
    chunks = (5, 10)
    shape = (5, 20)
    dtype = np.dtype("int32")
    zarray = ZArray(
        chunks=chunks,
        compressor={"id": "zlib", "level": 1},
        dtype=dtype,
        fill_value=0.0,
        filters=None,
        order="C",
        shape=shape,
        zarr_format=2,
    )

    chunks_dict = {
        "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds2 = xr.Dataset({"a": (["x", "y"], marr2)})
        assert ds1.equals(ds2)

        chunks_dict3 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(1, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
        }
        manifest2 = ChunkManifest(entries=chunks_dict2)
        marr2 = ManifestArray(zarray=zarray, chunkmanifest=manifest2)
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(5, 20),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        ds1 = xr.Dataset({"a": (["x", "y"], marr1)})

        chunks_dict2 = {
            "0.0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "0.1": {"path": "/foo.nc", "offset": 400, "length": 100},
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
            "0.0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "1.0.0": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "1.0.1": {"path": "file:///foo.nc", "offset": 400, "length": 100},
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
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("int32"),
            fill_value=0.0,
            filters=None,
            order="C",
            shape=(20,),
            zarr_format=2,
        )

        chunks_dict1 = {
            "0": {"path": "/foo.nc", "offset": 100, "length": 100},
            "1": {"path": "/foo.nc", "offset": 200, "length": 100},
        }
        manifest1 = ChunkManifest(entries=chunks_dict1)
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest1)
        coords = xr.Coordinates({"t": (["t"], marr1)}, indexes={})
        ds1 = xr.Dataset(coords=coords)

        chunks_dict2 = {
            "0": {"path": "/foo.nc", "offset": 300, "length": 100},
            "1": {"path": "/foo.nc", "offset": 400, "length": 100},
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
            "0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
            "2": {"path": "file:///foo.nc", "offset": 300, "length": 100},
            "3": {"path": "file:///foo.nc", "offset": 400, "length": 100},
        }
        assert result.data.zarray.compressor == zarray.compressor
        assert result.data.zarray.filters == zarray.filters
        assert result.data.zarray.fill_value == zarray.fill_value
        assert result.data.zarray.order == zarray.order
        assert result.data.zarray.zarr_format == zarray.zarr_format


@requires_kerchunk
@pytest.mark.parametrize("hdf_backend", [None, HDFVirtualBackend])
class TestCombineUsingIndexes:
    def test_combine_by_coords(self, netcdf4_files_factory: Callable, hdf_backend):
        filepath1, filepath2 = netcdf4_files_factory()

        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds1 = open_virtual_dataset(filepath1, backend=hdf_backend)
        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds2 = open_virtual_dataset(filepath2, backend=hdf_backend)

        combined_vds = xr.combine_by_coords(
            [vds2, vds1],
        )

        assert combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing

    @pytest.mark.xfail(reason="Not yet implemented, see issue #18")
    def test_combine_by_coords_keeping_manifestarrays(self, netcdf4_files, hdf_backend):
        filepath1, filepath2 = netcdf4_files

        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds1 = open_virtual_dataset(filepath1, backend=hdf_backend)
        with pytest.warns(UserWarning, match="will create in-memory pandas indexes"):
            vds2 = open_virtual_dataset(filepath2, backend=hdf_backend)

        combined_vds = xr.combine_by_coords(
            [vds2, vds1],
        )

        assert isinstance(combined_vds["time"].data, ManifestArray)
        assert isinstance(combined_vds["lat"].data, ManifestArray)
        assert isinstance(combined_vds["lon"].data, ManifestArray)


@requires_kerchunk
@pytest.mark.parametrize("hdf_backend", [None, HDFVirtualBackend])
class TestRenamePaths:
    def test_rename_to_str(self, netcdf4_file, hdf_backend):
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)
        renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )

    def test_rename_using_function(self, netcdf4_file, hdf_backend):
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)

        def local_to_s3_url(old_local_path: str) -> str:
            from pathlib import Path

            new_s3_bucket_url = "s3://bucket/"

            filename = Path(old_local_path).name
            return str(new_s3_bucket_url + filename)

        renamed_vds = vds.virtualize.rename_paths(local_to_s3_url)
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )

    def test_invalid_type(self, netcdf4_file, hdf_backend):
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)

        with pytest.raises(TypeError):
            vds.virtualize.rename_paths(["file1.nc", "file2.nc"])

    def test_mixture_of_manifestarrays_and_numpy_arrays(
        self, netcdf4_file, hdf_backend
    ):
        vds = open_virtual_dataset(
            netcdf4_file,
            indexes={},
            loadable_variables=["lat", "lon"],
            backend=hdf_backend,
        )
        renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
        assert (
            renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
            == "s3://bucket/air.nc"
        )
        assert isinstance(renamed_vds["lat"].data, np.ndarray)
