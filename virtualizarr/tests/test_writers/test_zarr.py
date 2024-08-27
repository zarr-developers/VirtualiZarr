import json

import pytest
from xarray import Dataset
import xarray.testing as xrt
import numpy as np

from virtualizarr import ManifestArray

from virtualizarr.writers.zarr import dataset_to_zarr
from virtualizarr.manifests.manifest import ChunkManifest
from virtualizarr import open_virtual_dataset
from virtualizarr.backend import FileType
from virtualizarr.readers.zarr import metadata_from_zarr_json
from virtualizarr.writers.zarr import dataset_to_zarr


@pytest.fixture
def vds_with_manifest_arrays() -> Dataset:
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="test.nc", offset=6144, length=48)}
        ),
        zarray=dict(
            shape=(2, 3),
            dtype=np.dtype("<i8"),
            chunks=(2, 3),
            compressor={"id": "zlib", "level": 1},
            filters=None,
            fill_value=0,
            order="C",
            zarr_format=3,
        ),
    )
    return Dataset({"a": (["x", "y"], arr)}, attrs={"something": 0})


def isconfigurable(value: dict) -> bool:
    """
    Several metadata attributes in ZarrV3 use a dictionary with keys "name" : str and "configuration" : dict
    """
    return "name" in value and "configuration" in value


def test_zarr_v3_metadata_conformance(tmpdir, vds_with_manifest_arrays: Dataset):
    """
    Checks that the output metadata of an array variable conforms to this spec
    for the required attributes:
    https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#metadata
    """
    dataset_to_zarr(vds_with_manifest_arrays, tmpdir / "store.zarr")
    # read the a variable's metadata
    with open(tmpdir / "store.zarr/a/zarr.json", mode="r") as f:
        metadata = json.loads(f.read())
    assert metadata["zarr_format"] == 3
    assert metadata["node_type"] == "array"
    assert isinstance(metadata["shape"], list) and all(
        isinstance(dim, int) for dim in metadata["shape"]
    )
    assert isinstance(metadata["data_type"], str) or isconfigurable(
        metadata["data_type"]
    )
    assert isconfigurable(metadata["chunk_grid"])
    assert isconfigurable(metadata["chunk_key_encoding"])
    assert isinstance(metadata["fill_value"], (bool, int, float, str, list))
    assert (
        isinstance(metadata["codecs"], list)
        and len(metadata["codecs"]) > 1
        and all(isconfigurable(codec) for codec in metadata["codecs"])
    )


def test_zarr_v3_roundtrip(tmpdir, vds_with_manifest_arrays: Dataset):
    vds_with_manifest_arrays.virtualize.to_zarr(tmpdir / "store.zarr")
    roundtrip = open_virtual_dataset(
        tmpdir / "store.zarr", filetype=FileType.zarr_v3, indexes={}
    )

    xrt.assert_identical(roundtrip, vds_with_manifest_arrays)


def test_metadata_roundtrip(tmpdir, vds_with_manifest_arrays: Dataset):
    dataset_to_zarr(vds_with_manifest_arrays, tmpdir / "store.zarr")
    zarray, _, _ = metadata_from_zarr_json(tmpdir / "store.zarr/a/zarr.json")
    assert zarray == vds_with_manifest_arrays.a.data.zarray
