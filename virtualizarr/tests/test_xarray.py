from typing import Callable

import numpy as np
import pytest
import xarray as xr
from xarray import open_dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.backend import VirtualBackend
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import (
    parametrize_over_hdf_backends,
    requires_hdf5plugin,
    requires_imagecodecs,
)


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
@parametrize_over_hdf_backends
class TestCombineUsingIndexes:
    def test_combine_by_coords(
        self,
        netcdf4_files_factory: Callable[[], tuple[str, str]],
        hdf_backend: type[VirtualBackend],
    ):
        filepath1, filepath2 = netcdf4_files_factory()

        with (
            open_virtual_dataset(
                filepath1,
                backend=hdf_backend,
                loadable_variables=["time", "lat", "lon"],
            ) as vds1,
            open_virtual_dataset(
                filepath2,
                backend=hdf_backend,
                loadable_variables=["time", "lat", "lon"],
            ) as vds2,
        ):
            combined_vds = xr.combine_by_coords(
                [vds2, vds1],
            )

            assert (
                combined_vds.xindexes["time"].to_pandas_index().is_monotonic_increasing
            )

    @pytest.mark.xfail(reason="Not yet implemented, see issue #18")
    def test_combine_by_coords_keeping_manifestarrays(
        self,
        netcdf4_files_factory: Callable[[], tuple[str, str]],
        hdf_backend: type[VirtualBackend],
    ):
        filepath1, filepath2 = netcdf4_files_factory()

        with (
            open_virtual_dataset(filepath1, backend=hdf_backend) as vds1,
            open_virtual_dataset(filepath2, backend=hdf_backend) as vds2,
        ):
            combined_vds = xr.combine_by_coords([vds2, vds1])

            assert isinstance(combined_vds["time"].data, ManifestArray)
            assert isinstance(combined_vds["lat"].data, ManifestArray)
            assert isinstance(combined_vds["lon"].data, ManifestArray)


@parametrize_over_hdf_backends
class TestRenamePaths:
    def test_rename_to_str(self, netcdf4_file, hdf_backend):
        with open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend) as vds:
            renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )

    def test_rename_using_function(self, netcdf4_file, hdf_backend):
        def local_to_s3_url(old_local_path: str) -> str:
            from pathlib import Path

            new_s3_bucket_url = "s3://bucket/"
            filename = Path(old_local_path).name
            return str(new_s3_bucket_url + filename)

        with open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend) as vds:
            renamed_vds = vds.virtualize.rename_paths(local_to_s3_url)
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )

    def test_invalid_type(self, netcdf4_file, hdf_backend):
        with open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend) as vds:
            with pytest.raises(TypeError):
                vds.virtualize.rename_paths(["file1.nc", "file2.nc"])

    @requires_hdf5plugin
    @requires_imagecodecs
    def test_mixture_of_manifestarrays_and_numpy_arrays(
        self, netcdf4_file, hdf_backend
    ):
        with open_virtual_dataset(
            netcdf4_file,
            indexes={},
            loadable_variables=["lat", "lon"],
            backend=hdf_backend,
        ) as vds:
            renamed_vds = vds.virtualize.rename_paths("s3://bucket/air.nc")
            assert (
                renamed_vds["air"].data.manifest.dict()["0.0.0"]["path"]
                == "s3://bucket/air.nc"
            )
            assert isinstance(renamed_vds["lat"].data, np.ndarray)


@requires_hdf5plugin
@requires_imagecodecs
def test_nbytes(simple_netcdf4):
    with open_virtual_dataset(simple_netcdf4) as vds:
        assert vds.virtualize.nbytes == 32
        assert vds.nbytes == 48

    with open_virtual_dataset(simple_netcdf4, loadable_variables=["foo"]) as vds:
        assert vds.virtualize.nbytes == 48

    with open_dataset(simple_netcdf4) as ds:
        assert ds.virtualize.nbytes == ds.nbytes
