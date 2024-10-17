import numpy as np
import xarray as xr
import xarray.testing as xrt

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.readers.kerchunk import (
    dataset_from_kerchunk_refs,
    find_var_names,
)
from virtualizarr.tests import requires_kerchunk


@requires_kerchunk
def test_kerchunk_roundtrip_in_memory_no_concat():
    # Set up example xarray dataset
    chunks_dict = {
        "0.0": {"path": "foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(
        zarray=dict(
            shape=(2, 4),
            dtype=np.dtype("<i8"),
            chunks=(2, 2),
            compressor=None,
            filters=None,
            fill_value=np.nan,
            order="C",
        ),
        chunkmanifest=manifest,
    )
    ds = xr.Dataset({"a": (["x", "y"], marr)})

    # Use accessor to write it out to kerchunk reference dict
    ds_refs = ds.virtualize.to_kerchunk(format="dict")

    # Use dataset_from_kerchunk_refs to reconstruct the dataset
    roundtrip = dataset_from_kerchunk_refs(ds_refs)

    # Assert equal to original dataset
    xrt.assert_equal(roundtrip, ds)


@requires_kerchunk
def test_no_duplicates_find_var_names():
    """Verify that we get a deduplicated list of var names"""
    ref_dict = {"refs": {"x/something": {}, "x/otherthing": {}}}
    assert len(find_var_names(ref_dict)) == 1
