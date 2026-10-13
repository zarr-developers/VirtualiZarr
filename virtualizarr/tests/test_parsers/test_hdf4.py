from pathlib import Path

import numpy as np
import pytest
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDF4Parser
from virtualizarr.tests import requires_kerchunk

DATA = Path(__file__).resolve().parents[1] / "data"

# A small MODIS fire-mask granule. Vendored from the kerchunk test suite so this
# stays a hermetic, offline test. This granule detected no fires, so all of its
# fire-pixel variables are zero-length and none of its data is contiguous.
FIXTURE = DATA / "MOD14.hdf4"

# Covers what MOD14.hdf4 cannot: a fire-pixel variable that actually holds data,
# stored as a single contiguous deflate-compressed block rather than as chunks.
# Real MOD14 granules with fires in them are behind an Earthdata login, so this is
# a minimal stand-in mimicking their structure, generated with:
#
#     from pyhdf.SD import SD, SDC
#     sd = SD("synthetic_fires.hdf4", SDC.WRITE | SDC.CREATE | SDC.TRUNC)
#     v = sd.create("FP_line", SDC.INT16, (4,))
#     v.setcompress(SDC.COMP_DEFLATE, 6)
#     v[:] = np.array([11, 22, 33, 44], dtype=">i2")
#     v.number_of_active_fires = 4
#     v.long_name = "granule line of fire pixel"
#     v.endaccess()
#     w = sd.create("fire mask", SDC.UINT8, (6, 4))
#     w.setcompress(SDC.COMP_DEFLATE, 6)
#     w[:] = np.arange(24, dtype="u1").reshape(6, 4)
#     w.endaccess()
#     sd.end()
FIRES_FIXTURE = DATA / "synthetic_fires.hdf4"


def _registry_and_url(fixture: Path = FIXTURE) -> tuple[ObjectStoreRegistry, str]:
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return registry, fixture.as_uri()


@requires_kerchunk
def test_hdf4_virtual_dataset() -> None:
    registry, url = _registry_and_url()
    parser = HDF4Parser()
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, Dataset)
        assert "fire mask" in vds.variables
        var = vds["fire mask"].variable
        assert var.sizes == {"fire mask_x": 2030, "fire mask_y": 1354}
        assert var.dtype == "uint8"


@requires_kerchunk
def test_hdf4_skip_variables() -> None:
    registry, url = _registry_and_url()
    parser = HDF4Parser(skip_variables=["algorithm QA"])
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert "fire mask" in vds.variables
        assert "algorithm QA" not in vds.variables


@requires_kerchunk
def test_hdf4_chunk_decodes_via_codec() -> None:
    """Read the "fire mask" array back through its byte references and zlib
    codec, asserting the exact max value the kerchunk test suite checks for."""
    registry, url = _registry_and_url()
    store = HDF4Parser()(url, registry)
    arr = zarr.open_array(store, path="fire mask", mode="r")
    assert np.asarray(arr[:]).max() == 5


@requires_kerchunk
def test_hdf4_zero_length_variables() -> None:
    """This granule detected no fires, so every `FP_*` fire-pixel variable has
    shape (0,). HDF4 reports a chunk edge of 0 for them, which Zarr rejects, so
    the parser coerces the edge to 1 without altering the (empty) shape."""
    registry, url = _registry_and_url()
    store = HDF4Parser()(url, registry)
    marr = store._group.arrays["FP_line"]
    assert marr.shape == (0,)
    assert marr.metadata.chunks == (1,)
    assert marr.manifest.shape_chunk_grid == (0,)
    assert marr.manifest.dict() == {}


@requires_kerchunk
def test_hdf4_contiguous_deflate_variable_decodes() -> None:
    """A fire-pixel variable holding real data is stored as one contiguous
    deflate-compressed block, not as chunks, so its whole array is a single
    chunk reference."""
    registry, url = _registry_and_url(FIRES_FIXTURE)
    store = HDF4Parser()(url, registry)
    marr = store._group.arrays["FP_line"]
    assert marr.shape == (4,)
    assert marr.metadata.chunks == (4,)
    assert marr.manifest.shape_chunk_grid == (1,)

    arr = zarr.open_array(store, path="FP_line", mode="r")
    np.testing.assert_array_equal(np.asarray(arr[:]), [11, 22, 33, 44])


@requires_kerchunk
@pytest.mark.parametrize(
    "fixture, name", [(FIXTURE, "fire mask"), (FIRES_FIXTURE, "FP_line")]
)
def test_hdf4_chunk_shape_is_not_leaked_into_attributes(
    fixture: Path, name: str
) -> None:
    """A variable's chunk shape is metadata, not one of its HDF4 attributes."""
    registry, url = _registry_and_url(fixture)
    store = HDF4Parser()(url, registry)
    assert "chunks" not in store._group.arrays[name].metadata.attributes
