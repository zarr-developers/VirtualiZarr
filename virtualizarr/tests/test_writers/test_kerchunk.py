import numpy as np
import pandas as pd
from xarray import Dataset

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import requires_kerchunk


@requires_kerchunk
class TestAccessor:
    def test_accessor_to_kerchunk_dict(self):
        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=np.nan,
                order="C",
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"dtype":"<i8","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }

        result_ds_refs = ds.virtualize.to_kerchunk(format="dict")
        assert result_ds_refs == expected_ds_refs

    def test_accessor_to_kerchunk_dict_empty(self):
        manifest = ChunkManifest(entries={}, shape=(1, 1))
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=np.nan,
                order="C",
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"dtype":"<i8","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            },
        }

        result_ds_refs = ds.virtualize.to_kerchunk(format="dict")
        assert result_ds_refs == expected_ds_refs

    def test_accessor_to_kerchunk_json(self, tmp_path):
        import ujson

        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=None,
                filters=None,
                fill_value=np.nan,
                order="C",
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"

        ds.virtualize.to_kerchunk(filepath, format="json")

        with open(filepath) as json_file:
            loaded_refs = ujson.load(json_file)

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"dtype":"<i8","fill_value":null,"order":"C","compressor":null,"filters":null,"zarr_format":2}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }
        assert loaded_refs == expected_ds_refs

    def test_accessor_to_kerchunk_parquet(self, tmp_path):
        import ujson

        chunks_dict = {
            "0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        arr = ManifestArray(
            chunkmanifest=manifest,
            zarray=dict(
                shape=(2, 4),
                dtype=np.dtype("<i8"),
                chunks=(2, 2),
                compressor=None,
                filters=None,
                fill_value=None,
                order="C",
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs"

        ds.virtualize.to_kerchunk(filepath, format="parquet", record_size=2)

        with open(tmp_path / "refs" / ".zmetadata") as f:
            meta = ujson.load(f)
            assert list(meta) == ["metadata", "record_size"]
            assert meta["record_size"] == 2

        df0 = pd.read_parquet(filepath / "a" / "refs.0.parq")

        assert df0.to_dict() == {
            "offset": {0: 100, 1: 200},
            "path": {
                0: "/foo.nc",
                1: "/foo.nc",
            },
            "size": {0: 100, 1: 100},
            "raw": {0: None, 1: None},
        }
