from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import pytest
import ujson

from virtualizarr.backend import open_virtual_dataset
from virtualizarr.manifests import ManifestArray


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
        zattrs = '{"_ARRAY_DIMENSIONS":["x","y"]}'
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
        filepath = tmp_path / "refs.json"

        with open(filepath, "w") as json_file:
            ujson.dump(refs, json_file)

        return str(filepath)

    yield _refs_file


def test_dataset_from_df_refs(refs_file_factory):
    refs_file = refs_file_factory()

    vds = open_virtual_dataset(refs_file, filetype="kerchunk")

    assert "a" in vds
    vda = vds["a"]
    assert isinstance(vda.data, ManifestArray)
    assert vda.dims == ("x", "y")
    assert vda.shape == (2, 3)
    assert vda.chunks == (2, 3)
    assert vda.dtype == np.dtype("<i8")

    assert vda.data.zarray.compressor is None
    assert vda.data.zarray.filters is None
    assert vda.data.zarray.fill_value == 0
    assert vda.data.zarray.order == "C"

    assert vda.data.manifest.dict() == {
        "0.0": {"path": "file:///test1.nc", "offset": 6144, "length": 48}
    }


def test_dataset_from_df_refs_with_filters(refs_file_factory):
    filters = [{"elementsize": 4, "id": "shuffle"}, {"id": "zlib", "level": 4}]
    zarray = {
        "chunks": [2, 3],
        "compressor": None,
        "dtype": "<i8",
        "fill_value": None,
        "filters": filters,
        "order": "C",
        "shape": [2, 3],
        "zarr_format": 2,
    }
    refs_file = refs_file_factory(zarray=ujson.dumps(zarray))

    vds = open_virtual_dataset(refs_file, filetype="kerchunk")

    vda = vds["a"]
    assert vda.data.zarray.filters == filters


def test_empty_chunk_manifest(refs_file_factory):
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

    vds = open_virtual_dataset(refs_file, filetype="kerchunk")

    assert "a" in vds.variables
    assert isinstance(vds["a"].data, ManifestArray)
    assert vds["a"].sizes == {"x": 100, "y": 200}
    assert vds["a"].chunksizes == {"x": 50, "y": 100}


def test_handle_relative_paths(refs_file_factory):
    # deliberately use relative path here, see https://github.com/zarr-developers/VirtualiZarr/pull/243#issuecomment-2492341326
    refs_file = refs_file_factory(chunks={"a/0.0": ["test1.nc", 6144, 48]})

    with pytest.raises(ValueError, match="must be absolute posix paths"):
        open_virtual_dataset(refs_file, filetype="kerchunk")

    refs_file = refs_file_factory(chunks={"a/0.0": ["./test1.nc", 6144, 48]})
    with pytest.raises(ValueError, match="must be absolute posix paths"):
        open_virtual_dataset(refs_file, filetype="kerchunk")

    with pytest.raises(
        ValueError, match="fs_root must be an absolute path to a filesystem directory"
    ):
        open_virtual_dataset(
            refs_file,
            filetype="kerchunk",
            virtual_backend_kwargs={"fs_root": "some_directory/"},
        )

    with pytest.raises(
        ValueError, match="fs_root must be an absolute path to a filesystem directory"
    ):
        open_virtual_dataset(
            refs_file,
            filetype="kerchunk",
            virtual_backend_kwargs={"fs_root": "/some_directory/file.nc"},
        )

    vds = open_virtual_dataset(
        refs_file,
        filetype="kerchunk",
        virtual_backend_kwargs={"fs_root": "/some_directory/"},
    )
    vda = vds["a"]
    assert vda.data.manifest.dict() == {
        "0.0": {"path": "file:///some_directory/test1.nc", "offset": 6144, "length": 48}
    }

    vds = open_virtual_dataset(
        refs_file,
        filetype="kerchunk",
        virtual_backend_kwargs={"fs_root": "file:///some_directory/"},
    )
    vda = vds["a"]
    assert vda.data.manifest.dict() == {
        "0.0": {"path": "file:///some_directory/test1.nc", "offset": 6144, "length": 48}
    }
