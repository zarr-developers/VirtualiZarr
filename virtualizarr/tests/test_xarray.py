import numpy as np
import xarray as xr

from virtualizarr.manifests import ChunkManifest, ManifestArray
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

        # TODO this fails because xr.concat tries to rebuild indexes on dimension coordinates automatically
        # should there be a join='ignore' option?
        result = xr.concat([ds1, ds2], dim="t", coords="minimal", compat="override")[
            "a"
        ]

        # TODO this also causes a separate problem
        # TypeError: Could not find a Chunk Manager which recognises type <class 'virtualizarr.manifests.array.ManifestArray'>
        # result = xr.concat([ds1, ds2], dim="x")["a"]

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
