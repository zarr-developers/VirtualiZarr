from pathlib import Path

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDF4Parser
from virtualizarr.tests import requires_kerchunk

# A small MODIS fire-mask granule. Vendored from the kerchunk test suite so this
# stays a hermetic, offline test.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "MOD14.hdf4"


def _registry_and_url() -> tuple[ObjectStoreRegistry, str]:
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return registry, FIXTURE.as_uri()


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
