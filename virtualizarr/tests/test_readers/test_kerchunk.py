from pathlib import Path
from typing import Any, Callable, Generator, Optional

import numpy as np
import pytest
import ujson

from virtualizarr.backend import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.readers.kerchunk import (
    dataset_from_kerchunk_refs,
)


def gen_ds_refs(
    zgroup: str | None = None,
    zarray: str | None = None,
    zattrs: str | None = None,
    chunk: list | None = None,
):
    if zgroup is None:
        zgroup = '{"zarr_format":2}'
    if zarray is None:
        zarray = '{"chunks":[2,3],"compressor":null,"dtype":"<i8","fill_value":null,"filters":null,"order":"C","shape":[2,3],"zarr_format":2}'
    if zattrs is None:
        zattrs = '{"_ARRAY_DIMENSIONS":["x","y"]}'
    if chunk is None:
        chunk = ["test1.nc", 6144, 48]

    return {
        "version": 1,
        "refs": {
            ".zgroup": zgroup,
            "a/.zarray": zarray,
            "a/.zattrs": zattrs,
            "a/0.0": chunk,
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

    def _refs_file(zgroup=None, zarray=None, zattrs=None, chunk=None) -> str:
        refs = gen_ds_refs(zgroup=zgroup, zarray=zarray, zattrs=zattrs, chunk=chunk)
        filepath = tmp_path / "refs.json"

        with open(filepath, "w") as json_file:
            ujson.dump(refs, json_file)

        return str(filepath)

    yield _refs_file


def test_dataset_from_df_refs(refs_file_factory: str):
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
        "0.0": {"path": "test1.nc", "offset": 6144, "length": 48}
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


def test_dataset_from_kerchunk_refs_empty_chunk_manifest():
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
    refs = gen_ds_refs(zarray=ujson.dumps(zarray))
    del refs["refs"]["a/0.0"]

    ds = dataset_from_kerchunk_refs(refs)
    assert "a" in ds.variables
    assert isinstance(ds["a"].data, ManifestArray)
    assert ds["a"].sizes == {"x": 100, "y": 200}
    assert ds["a"].chunksizes == {"x": 50, "y": 100}
