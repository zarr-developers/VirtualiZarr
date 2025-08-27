import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import Dataset
from zarr.core.metadata.v2 import ArrayV2Metadata

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import requires_fastparquet, requires_kerchunk
from virtualizarr.utils import JSON, convert_v3_to_v2_metadata, kerchunk_refs_as_json


def test_deserialize_to_json():
    refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": "{}",
            "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": ["/test.nc", 6144, 48],
        },
    }
    json_expected: JSON = {
        "version": 1,
        "refs": {
            ".zgroup": {"zarr_format": 2},
            ".zattrs": {},
            "a/.zarray": {
                "shape": [2, 3],
                "chunks": [2, 3],
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
                "compressor": None,
                "attributes": {},
                "zarr_format": 2,
                "dtype": "<i8",
            },
            "a/.zattrs": {
                "_ARRAY_DIMENSIONS": ["x", "y"],
            },
            "a/0.0": ["/test.nc", 6144, 48],
        },
    }
    refs_as_json = kerchunk_refs_as_json(refs)
    assert refs_as_json == json_expected


@requires_kerchunk
class TestAccessor:
    def test_accessor_to_kerchunk_dict(self, array_v3_metadata):
        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }

        result_ds_refs = ds.vz.to_kerchunk(format="dict")
        assert kerchunk_refs_as_json(result_ds_refs) == kerchunk_refs_as_json(
            expected_ds_refs
        )

    def test_accessor_to_kerchunk_dict_empty(self, array_v3_metadata):
        manifest = ChunkManifest(entries={}, shape=(1, 1))
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            },
        }

        result_ds_refs = ds.vz.to_kerchunk(format="dict")
        assert kerchunk_refs_as_json(result_ds_refs) == kerchunk_refs_as_json(
            expected_ds_refs
        )

    def test_accessor_to_kerchunk_json(self, tmp_path, array_v3_metadata):
        import ujson

        manifest = ChunkManifest(
            entries={"0.0": dict(path="file:///test.nc", offset=6144, length=48)}
        )
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 3),
                data_type=np.dtype("<i8"),
                chunks=(2, 3),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs.json"

        ds.vz.to_kerchunk(filepath, format="json")

        with open(filepath) as json_file:
            loaded_refs = ujson.load(json_file)

        expected_ds_refs = {
            "version": 1,
            "refs": {
                ".zgroup": '{"zarr_format":2}',
                ".zattrs": "{}",
                "a/.zarray": '{"shape":[2,3],"chunks":[2,3],"fill_value":0,"order":"C","filters":null,"dimension_separator":".","compressor":null,"attributes":{},"zarr_format":2,"dtype":"<i8"}',
                "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
                "a/0.0": ["/test.nc", 6144, 48],
            },
        }
        assert kerchunk_refs_as_json(loaded_refs) == kerchunk_refs_as_json(
            expected_ds_refs
        )

    @requires_fastparquet
    def test_accessor_to_kerchunk_parquet(self, tmp_path, array_v3_metadata):
        import ujson

        chunks_dict = {
            "0.0": {"path": "file:///foo.nc", "offset": 100, "length": 100},
            "0.1": {"path": "file:///foo.nc", "offset": 200, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        arr = ManifestArray(
            chunkmanifest=manifest,
            metadata=array_v3_metadata(
                shape=(2, 4),
                data_type=np.dtype("<i8"),
                chunks=(2, 2),
                codecs=[],
                fill_value=None,
            ),
        )
        ds = Dataset({"a": (["x", "y"], arr)})

        filepath = tmp_path / "refs"

        ds.vz.to_kerchunk(filepath, format="parquet", record_size=2)

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


@pytest.mark.parametrize(
    ["dtype", "endian", "expected_dtype_char"],
    [("i8", "little", "<"), ("i8", "big", ">"), ("i1", None, "|")],
)
def test_convert_v3_to_v2_metadata(
    array_v3_metadata, dtype: str, endian: str | None, expected_dtype_char: str
):
    shape = (5, 20)
    chunks = (5, 10)

    codecs = [
        {"name": "bytes", "configuration": {"endian": endian}},
        {
            "name": "numcodecs.delta",
            "configuration": {"dtype": f"{expected_dtype_char}{dtype}"},
        },
        {
            "name": "numcodecs.blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": 1},
        },
    ]

    v3_metadata = array_v3_metadata(
        data_type=np.dtype(dtype), shape=shape, chunks=chunks, codecs=codecs
    )
    v2_metadata = convert_v3_to_v2_metadata(v3_metadata)

    assert isinstance(v2_metadata, ArrayV2Metadata)
    assert v2_metadata.shape == shape
    expected_dtype = np.dtype(f"{expected_dtype_char}{dtype}")
    assert v2_metadata.dtype.to_native_dtype() == expected_dtype
    assert v2_metadata.chunks == chunks
    assert v2_metadata.fill_value == 0

    assert v2_metadata.filters
    filter_codec, compressor_codec = v2_metadata.filters
    compressor_config = compressor_codec.get_config()
    assert compressor_config["id"] == "blosc"
    assert compressor_config["cname"] == "zstd"
    assert compressor_config["clevel"] == 5
    assert compressor_config["shuffle"] == 1
    assert compressor_config["blocksize"] == 0

    filters_config = filter_codec.get_config()
    assert filters_config["id"] == "delta"
    expected_delta_dtype = f"{expected_dtype_char}{dtype}"
    assert filters_config["dtype"] == expected_delta_dtype
    assert filters_config["astype"] == expected_delta_dtype
    assert v2_metadata.attributes == {}


def test_warn_if_no_virtual_vars():
    non_virtual_ds = xr.Dataset({"foo": ("x", [10, 20, 30]), "x": ("x", [1, 2, 3])})
    with pytest.warns(UserWarning, match="non-virtual"):
        non_virtual_ds.vz.to_kerchunk()
