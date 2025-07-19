from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import obstore
import pytest
import ujson
import xarray as xr
import xarray.testing as xrt

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestStore,
)
from virtualizarr.parsers import KerchunkJSONParser, KerchunkParquetParser
from virtualizarr.registry import ObjectStoreRegistry, UrlKey
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
        ValueError, match="fs_root must be an absolute path to a filesystem directory"
    ):
        with open_virtual_dataset(
            url=refs_file,
            registry=local_registry,
            parser=parser,
        ) as _:
            pass

    parser = KerchunkJSONParser(fs_root="/some_directory/file.nc")
    with pytest.raises(
        ValueError, match="fs_root must be an absolute path to a filesystem directory"
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
            from kerchunk.df import refs_to_dataframe

            ref_filepath = tmp_path / "ref.parquet"
            refs_to_dataframe(fo=example_reference_dict, url=ref_filepath.as_posix())
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
def test_notimplemented_read_inline_refs(tmp_path, netcdf4_inlined_ref, local_registry):
    # For now, we raise a NotImplementedError if we read existing references that have inlined data
    # https://github.com/zarr-developers/VirtualiZarr/pull/251#pullrequestreview-2361916932

    ref_filepath = tmp_path / "ref.json"

    import ujson

    with open(ref_filepath, "w") as json_file:
        ujson.dump(netcdf4_inlined_ref, json_file)

    parser = KerchunkJSONParser()
    with pytest.raises(
        NotImplementedError,
        match="Reading inlined reference data is currently not supported",
    ):
        with open_virtual_dataset(
            url=ref_filepath.as_posix(),
            registry=local_registry,
            parser=parser,
        ) as _:
            pass


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
    refs: dict = gen_ds_refs()

    memory_store = obstore.store.MemoryStore()
    memory_store.put("refs.json", ujson.dumps(refs).encode())

    registry = ObjectStoreRegistry({"memory://": memory_store})
    parser = KerchunkJSONParser()
    manifeststore = parser("memory:///refs.json", registry=registry)

    assert isinstance(manifeststore, ManifestStore)
    assert manifeststore._registry.map[UrlKey("memory", "")].store == memory_store

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
