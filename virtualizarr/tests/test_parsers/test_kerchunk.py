from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import obstore
import pytest
import ujson
import xarray as xr
import xarray.testing as xrt
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestStore,
)
from virtualizarr.parsers import KerchunkJSONParser, KerchunkParquetParser
from virtualizarr.tests import has_fastparquet, requires_kerchunk
from virtualizarr.xarray import open_virtual_dataset


def gen_ds_refs(
    zgroup: str | None = None,
    zarray: str | None = None,
    zattrs: str | None = None,
    chunks: dict[str, list[str | int]] | None = None,
):
    if zgroup is None:
        zgroup = '{"zarr_format":2}'
    if zarray is None:
        zarray = '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}'
    if zattrs is None:
        zattrs = '{"_ARRAY_DIMENSIONS":["x","y"],"value": "1"}'
    if chunks is None:
        chunks = {"a/0.0": ["/test1.nc", 6144, 48]}

    return {
        "version": 1,
        "refs": {
            ".zgroup": zgroup,
            "a/.zarray": zarray,
            "a/.zattrs": zattrs,
            **chunks,
        },
    }


@pytest.fixture
def refs_file_factory(
    tmp_path: Path,
) -> Generator[
    Callable[[Optional[Any], Optional[Any], Optional[Any], Optional[Any]], str],
    None,
    None,
]:
    """
    Fixture which defers creation of the references file until the parameters zgroup etc. are known.
    """

    def _refs_file(zgroup=None, zarray=None, zattrs=None, chunks=None) -> str:
        refs = gen_ds_refs(zgroup=zgroup, zarray=zarray, zattrs=zattrs, chunks=chunks)
        url = tmp_path / "refs.json"

        with open(url, "w") as json_file:
            ujson.dump(refs, json_file)

        return str(url)

    yield _refs_file


def test_dataset_from_df_refs(refs_file_factory, local_registry):
    refs_file = refs_file_factory()
    refs_url = f"file://{refs_file}"
    parser = KerchunkJSONParser()
    with open_virtual_dataset(
        url=refs_url, registry=local_registry, parser=parser
    ) as vds:
        assert "a" in vds
        vda = vds["a"]
        assert isinstance(vda.data, ManifestArray)
        assert vda.dims == ("x", "y")
        assert vda.shape == (2, 3)
        assert vda.chunks == (2, 3)
        assert vda.dtype == np.dtype("<i8")
        assert vda.attrs == {"value": "1"}

        assert vda.data.metadata.codecs[0].to_dict() == {
            "configuration": {"endian": "little"},
            "name": "bytes",
        }
        assert vda.data.metadata.fill_value == 0

        assert vda.data.manifest.dict() == {
            "0.0": {"path": "file:///test1.nc", "offset": 6144, "length": 48}
        }


def test_dataset_from_df_refs_with_filters(refs_file_factory, local_registry):
    compressor = [{"elementsize": 4, "id": "shuffle"}, {"id": "zlib", "level": 4}]
    zarray = {
        "chunks": [2, 3],
        "compressor": compressor,
        "dtype": "<i8",
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": [2, 3],
        "zarr_format": 2,
    }
    refs_file = refs_file_factory(zarray=ujson.dumps(zarray))
    parser = KerchunkJSONParser(fs_root="file://")
    with open_virtual_dataset(
        url=refs_file, registry=local_registry, parser=parser
    ) as vds:
        vda = vds["a"]
        assert vda.data.metadata.codecs[1].to_dict() == {
            "name": "numcodecs.shuffle",
            "configuration": {"elementsize": 4},
        }


def test_empty_chunk_manifest(refs_file_factory, local_registry):
    zarray = {
        "chunks": [50, 100],
        "compressor": None,
        "dtype": "<i8",
        "fill_value": 100,
        "filters": None,
        "order": "C",
        "shape": [100, 200],
        "zarr_format": 2,
    }
    refs_file = refs_file_factory(zarray=ujson.dumps(zarray), chunks={})
    parser = KerchunkJSONParser()
    with open_virtual_dataset(
        url=refs_file, registry=local_registry, parser=parser
    ) as vds:
        assert "a" in vds.variables
        assert isinstance(vds["a"].data, ManifestArray)
        assert vds["a"].sizes == {"x": 100, "y": 200}
        assert vds["a"].chunksizes == {"x": 50, "y": 100}


def test_null_chunk_reference_treated_as_missing():
    """Test that a kerchunk chunk reference containing NaN (from JSON null) is treated as missing."""
    from virtualizarr.parsers.kerchunk.translator import chunkentry_from_kerchunk

    # In kerchunk parquet format, a JSON null in the chunk reference deserializes to NaN
    # This should be interpreted as a missing/uninitialized chunk
    chunk_entry = chunkentry_from_kerchunk([float("nan")])

    assert chunk_entry["path"] == ""
    assert chunk_entry["offset"] == 0
    assert chunk_entry["length"] == 0


@requires_kerchunk
@pytest.mark.skipif(not has_fastparquet, reason="fastparquet not installed")
def test_kerchunk_parquet_sparse_array(tmp_path, local_registry):
    """
    Integration test: kerchunk parquet with sparse chunks (some missing) should work.

    This tests reading a kerchunk parquet where not all chunks are present,
    which is a common case for sparse arrays.
    """
    import pandas as pd
    from kerchunk.df import refs_to_dataframe

    # Create refs with only one chunk defined (sparse array)
    refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            "a/.zarray": '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":0,"filters":null,"order":"C","shape":[4,3],"zarr_format":2}',
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": ["/test1.nc", 6144, 48],
            # a/1.0 is intentionally missing - sparse array
        },
    }

    ref_filepath = tmp_path / "sparse.parq"
    with pd.option_context("future.infer_string", False):
        refs_to_dataframe(fo=refs, url=str(ref_filepath))

    parser = KerchunkParquetParser()
    with open_virtual_dataset(
        url=str(ref_filepath),
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert "a" in vds.variables
        manifest = vds["a"].data.manifest.dict()
        # Chunk 0.0 should have valid reference
        assert manifest["0.0"]["path"] == "file:///test1.nc"
        assert manifest["0.0"]["offset"] == 6144
        assert manifest["0.0"]["length"] == 48
        # Chunk 1.0 is not in manifest (sparse array - missing chunks omitted)
        assert "1.0" not in manifest


def test_scalar_variable_manifest_shape(refs_file_factory, local_registry):
    """Scalar variables from kerchunk refs should produce a manifest with shape (), not (1,)."""
    zarray = '{"chunks":[],"compressor":null,"dtype":"<i4","fill_value":null,"filters":null,"order":"C","shape":[],"zarr_format":2}'
    zattrs = '{"_ARRAY_DIMENSIONS":[]}'
    refs_file = refs_file_factory(
        zarray=zarray,
        zattrs=zattrs,
        chunks={"a/0": ["/test1.nc", 100, 4]},
    )
    parser = KerchunkJSONParser()
    with open_virtual_dataset(
        url=refs_file, registry=local_registry, parser=parser
    ) as vds:
        assert vds["a"].shape == ()
        assert vds["a"].data.manifest.shape_chunk_grid == ()


def test_handle_relative_paths(refs_file_factory, local_registry):
    # deliberately use relative path here, see https://github.com/zarr-developers/VirtualiZarr/pull/243#issuecomment-2492341326
    refs_file = refs_file_factory(chunks={"a/0.0": ["test1.nc", 6144, 48]})
    parser = KerchunkJSONParser()
    with pytest.raises(ValueError, match="must be absolute posix paths"):
        with open_virtual_dataset(
            url=refs_file,
            registry=local_registry,
            parser=parser,
        ) as _:
            pass

    refs_file = refs_file_factory(chunks={"a/0.0": ["./test1.nc", 6144, 48]})
    parser = KerchunkJSONParser()
    with pytest.raises(ValueError, match="must be absolute posix paths"):
        with open_virtual_dataset(
            url=refs_file,
            registry=local_registry,
            parser=parser,
        ) as _:
            pass

    parser = KerchunkJSONParser(fs_root="some_directory/")
    with pytest.raises(
        ValueError, match="fs_root must be an absolute filesystem directory path or URI"
    ):
        with open_virtual_dataset(
            url=refs_file,
            registry=local_registry,
            parser=parser,
        ) as _:
            pass

    parser = KerchunkJSONParser(fs_root="/some_directory/file.nc")
    with pytest.raises(
        ValueError, match="fs_root must be an absolute filesystem directory path or URI"
    ):
        with open_virtual_dataset(
            url=refs_file,
            registry=local_registry,
            parser=parser,
        ) as _:
            pass
    parser = KerchunkJSONParser(fs_root="/some_directory/")
    with open_virtual_dataset(
        url=refs_file,
        registry=local_registry,
        parser=parser,
    ) as vds:
        vda = vds["a"]
        assert vda.data.manifest.dict() == {
            "0.0": {
                "path": "file:///some_directory/test1.nc",
                "offset": 6144,
                "length": 48,
            }
        }

    parser = KerchunkJSONParser(fs_root="file:///some_directory/")
    with open_virtual_dataset(
        url=refs_file,
        registry=local_registry,
        parser=parser,
    ) as vds:
        vda = vds["a"]
        assert vda.data.manifest.dict() == {
            "0.0": {
                "path": "file:///some_directory/test1.nc",
                "offset": 6144,
                "length": 48,
            }
        }


def test_relative_path_to_absolute_cloud_url(refs_file_factory, local_registry):
    # regression test for https://github.com/zarr-developers/VirtualiZarr/issues/975
    refs_file = refs_file_factory(chunks={"a/0.0": ["test1.nc", 6144, 48]})
    parser = KerchunkJSONParser(fs_root="s3://some_bucket/")
    with open_virtual_dataset(
        url=refs_file,
        registry=local_registry,
        parser=parser,
    ) as vds:
        vda = vds["a"]
        assert vda.data.manifest.dict() == {
            "0.0": {
                "path": "s3://some_bucket/test1.nc",
                "offset": 6144,
                "length": 48,
            }
        }


@requires_kerchunk
@pytest.mark.parametrize(
    "reference_format",
    ["json", "invalid", *(["parquet"] if has_fastparquet else [])],
)
def test_open_virtual_dataset_existing_kerchunk_refs(
    tmp_path, netcdf4_virtual_dataset, reference_format, local_registry
):
    example_reference_dict = netcdf4_virtual_dataset.vz.to_kerchunk(format="dict")

    if reference_format == "invalid":
        # Test invalid file format leads to ValueError
        ref_filepath = tmp_path / "ref.csv"
        with open(ref_filepath.as_posix(), mode="w") as of:
            of.write("tmp")
        parser = KerchunkJSONParser(fs_root="file://")
        with pytest.raises(ValueError):
            with open_virtual_dataset(
                url=ref_filepath.as_posix(),
                registry=local_registry,
                parser=parser,
            ) as _:
                pass
    else:
        # Test valid json and parquet reference formats
        if reference_format == "json":
            ref_filepath = tmp_path / "ref.json"

            import ujson

            with open(ref_filepath, "w") as json_file:
                ujson.dump(example_reference_dict, json_file)
            parser = KerchunkJSONParser(fs_root="file://")
        if reference_format == "parquet":
            import pandas as pd
            from kerchunk.df import refs_to_dataframe

            ref_filepath = tmp_path / "ref.parquet"
            with pd.option_context("future.infer_string", False):
                refs_to_dataframe(
                    fo=example_reference_dict, url=ref_filepath.as_posix()
                )
            parser = KerchunkParquetParser(fs_root="file://")
        expected_refs = netcdf4_virtual_dataset.vz.to_kerchunk(format="dict")
        with open_virtual_dataset(
            url=ref_filepath.as_posix(),
            registry=local_registry,
            parser=parser,
            loadable_variables=[],
        ) as vds:
            # Inconsistent results! https://github.com/zarr-developers/VirtualiZarr/pull/73#issuecomment-2040931202
            # assert vds.vz.to_kerchunk(format='dict') == example_reference_dict
            refs = vds.vz.to_kerchunk(format="dict")
            expected_refs = netcdf4_virtual_dataset.vz.to_kerchunk(format="dict")
            assert refs["refs"]["air/0.0.0"] == expected_refs["refs"]["air/0.0.0"]
            assert refs["refs"]["lon/0"] == expected_refs["refs"]["lon/0"]
            assert refs["refs"]["lat/0"] == expected_refs["refs"]["lat/0"]
            assert refs["refs"]["time/0"] == expected_refs["refs"]["time/0"]

            assert list(vds) == list(netcdf4_virtual_dataset)
            assert set(vds.coords) == set(netcdf4_virtual_dataset.coords)
            assert set(vds.variables) == set(netcdf4_virtual_dataset.variables)


@requires_kerchunk
def _refs_with_one_inlined_one_virtual(inline_repr: str) -> dict:
    """Kerchunk refs describing a (2, 4) int32 array with two (1, 4) chunks:
    position (0, 0) is inlined (carrying ``inline_repr``), position (1, 0) is virtual."""
    return {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            "a/.zarray": (
                '{"chunks":[1,4],"compressor":null,"dtype":"<i4",'
                '"fill_value":null,"filters":null,"order":"C",'
                '"shape":[2,4],"zarr_format":2}'
            ),
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": inline_repr,
            "a/1.0": ["/test.nc", 6144, 16],
        },
    }


@pytest.mark.parametrize(
    "inline_repr, expected_bytes",
    [
        ("base64:AQIDBAUGBwg=", b"\x01\x02\x03\x04\x05\x06\x07\x08"),
        ("hello, w!", b"hello, w!"),
    ],
    ids=["base64", "raw_string"],
)
def test_parse_inline_refs_json(inline_repr, expected_bytes):
    refs = _refs_with_one_inlined_one_virtual(inline_repr)

    memory_store = obstore.store.MemoryStore()
    memory_store.put("refs.json", ujson.dumps(refs).encode())
    registry = ObjectStoreRegistry({"memory://": memory_store})
    parser = KerchunkJSONParser()
    manifeststore = parser("memory:///refs.json", registry=registry)

    marr = manifeststore._group._members["a"]
    assert marr.manifest._inlined == {(0, 0): expected_bytes}
    assert marr.manifest.dict() == {
        "0.0": {
            "path": "__inlined__",
            "offset": 0,
            "length": len(expected_bytes),
            "data": expected_bytes,
        },
        "1.0": {"path": "file:///test.nc", "offset": 6144, "length": 16},
    }


@requires_kerchunk
@pytest.mark.skipif(not has_fastparquet, reason="fastparquet not installed")
def test_parse_inline_refs_parquet(tmp_path, local_registry):
    import pandas as pd
    from kerchunk.df import refs_to_dataframe

    expected_bytes = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    refs = _refs_with_one_inlined_one_virtual(
        "base64:" + "AQIDBAUGBwg="
    )

    ref_filepath = tmp_path / "ref.parquet"
    with pd.option_context("future.infer_string", False):
        refs_to_dataframe(fo=refs, url=ref_filepath.as_posix())

    parser = KerchunkParquetParser()
    manifeststore = parser(ref_filepath.as_posix(), registry=local_registry)

    marr = manifeststore._group._members["a"]
    assert marr.manifest._inlined == {(0, 0): expected_bytes}
    assert marr.manifest.dict() == {
        "0.0": {
            "path": "__inlined__",
            "offset": 0,
            "length": len(expected_bytes),
            "data": expected_bytes,
        },
        "1.0": {"path": "file:///test.nc", "offset": 6144, "length": 16},
    }


@pytest.mark.asyncio
async def test_read_inline_and_virtual_refs_end_to_end():
    # Parse a refs dict containing one inlined chunk and one virtual chunk,
    # then round-trip both through ManifestStore.get: the inlined chunk must
    # come back from memory and the virtual chunk must be fetched from the
    # target file via the object store registry.
    from zarr.core.buffer import default_buffer_prototype

    inlined_bytes = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    virtual_bytes = b"\xaa\xbb\xcc\xdd\xee\xff\x11\x22"

    refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            "a/.zarray": (
                '{"chunks":[1,4],"compressor":null,"dtype":"<i4",'
                '"fill_value":null,"filters":null,"order":"C",'
                '"shape":[2,4],"zarr_format":2}'
            ),
            "a/.zattrs": '{"_ARRAY_DIMENSIONS":["x","y"]}',
            "a/0.0": "base64:AQIDBAUGBwg=",
            "a/1.0": ["memory:///test.nc", 0, len(virtual_bytes)],
        },
    }

    memory_store = obstore.store.MemoryStore()
    memory_store.put("refs.json", ujson.dumps(refs).encode())
    memory_store.put("test.nc", virtual_bytes)

    registry = ObjectStoreRegistry({"memory://": memory_store})
    parser = KerchunkJSONParser()
    manifeststore = parser("memory:///refs.json", registry=registry)

    inlined_result = await manifeststore.get(
        "a/c/0/0", prototype=default_buffer_prototype()
    )
    assert inlined_result.to_bytes() == inlined_bytes

    virtual_result = await manifeststore.get(
        "a/c/1/0", prototype=default_buffer_prototype()
    )
    assert virtual_result.to_bytes() == virtual_bytes


@pytest.mark.parametrize("skip_variables", ["a", ["a"]])
def test_skip_variables(refs_file_factory, skip_variables, local_registry):
    refs_file = refs_file_factory()
    parser = KerchunkJSONParser(skip_variables=skip_variables, fs_root="file://")
    with open_virtual_dataset(
        url=refs_file,
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert all(var not in vds for var in skip_variables)


@requires_kerchunk
def test_load_manifest(tmp_path, netcdf4_file, netcdf4_virtual_dataset, local_registry):
    refs = netcdf4_virtual_dataset.vz.to_kerchunk(format="dict")
    ref_filepath = tmp_path / "ref.json"
    with open(ref_filepath.as_posix(), "w") as json_file:
        ujson.dump(refs, json_file)

    parser = KerchunkJSONParser()
    manifest_store = parser(
        url=f"file://{ref_filepath.as_posix()}", registry=local_registry
    )
    with (
        xr.open_dataset(
            netcdf4_file,
        ) as ds,
        xr.open_dataset(
            manifest_store,
            engine="zarr",
            consolidated=False,
            zarr_format=3,
        ).load() as manifest_ds,
    ):
        xrt.assert_identical(ds, manifest_ds)


def test_parse_dict_via_memorystore(array_v3_metadata):
    # generate some example kerchunk references
    refs = gen_ds_refs()

    memory_store = obstore.store.MemoryStore()
    memory_store.put("refs.json", ujson.dumps(refs).encode())

    registry = ObjectStoreRegistry({"memory://": memory_store})
    parser = KerchunkJSONParser()
    manifeststore = parser("memory:///refs.json", registry=registry)

    assert isinstance(manifeststore, ManifestStore)

    # assert metadata parsed correctly
    expected_metadata = array_v3_metadata(
        shape=(2, 3),
        chunks=(2, 3),
        data_type=np.dtype("int64"),
        dimension_names=["x", "y"],
        attributes={"value": "1"},
    )
    manifest = ChunkManifest(
        {
            "0": {"path": "/test1.nc", "offset": 6144, "length": 48},
        }
    )
    expected_marr = ManifestArray(
        metadata=expected_metadata,
        chunkmanifest=manifest,
    )
    # TODO this might be easier if `ManifestStore/Group` had __eq__ methods?
    actual_marr = manifeststore._group._members["a"]
    assert (actual_marr == expected_marr).all()

    # TODO assert that manifeststore.to_kerchunk_refs() roundtrips, once we have that method
