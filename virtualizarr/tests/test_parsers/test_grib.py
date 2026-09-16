from pathlib import Path

import numpy as np
import pytest
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import DataTree

from virtualizarr import open_virtual_datatree
from virtualizarr.tests import requires_grib

# The GribberishParser (and its zarr codec) are new in gribberish 1.0.0.
# From gribberish >=1.3.0 the parser groups messages by level-type and step-type
# (like cfgrib), so the soil variables live in a nested "/depth_bls/instant"
# group rather than at the store root. We open the whole hierarchy as a
# DataTree and assert against that layout.
pytest.importorskip("gribberish", minversion="1.3.0")

from gribberish.virtualizarr import GribberishParser  # noqa: E402

# A 1 KB GRIB1 file with 8 ECMWF soil variables on a 4x4 grid. Vendored from the
# gribberish test suite so this stays a hermetic, offline test.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "ecmwf_soil_8vars_tiny.grib1"

# The variables are soil layers (depth below land surface) at a single instant,
# so gribberish nests them under this group path.
SOIL_GROUP = "/depth_bls/instant"

# 4 soil-temperature layers + 4 soil-moisture layers.
EXPECTED_VARS = {"stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"}


def _registry_and_url() -> tuple[ObjectStoreRegistry, str]:
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return registry, FIXTURE.as_uri()


@requires_grib
def test_grib_virtual_datatree() -> None:
    registry, url = _registry_and_url()
    parser = GribberishParser()
    with open_virtual_datatree(url=url, parser=parser, registry=registry) as vdt:
        assert isinstance(vdt, DataTree)
        ds = vdt[SOIL_GROUP].ds
        assert set(ds.data_vars) == EXPECTED_VARS
        assert {"latitude", "longitude"}.issubset(set(ds.coords))
        for name in EXPECTED_VARS:
            var = ds[name]
            assert var.dims == ("time", "latitude", "longitude")
            assert var.sizes == {"time": 1, "latitude": 4, "longitude": 4}


@requires_grib
def test_grib_only_variables() -> None:
    registry, url = _registry_and_url()
    parser = GribberishParser(only_variables=["stl1"])
    with open_virtual_datatree(url=url, parser=parser, registry=registry) as vdt:
        assert set(vdt[SOIL_GROUP].ds.data_vars) == {"stl1"}


@requires_grib
def test_grib_chunk_decodes_via_codec() -> None:
    """Each GRIB message is one chunk; reading it back resolves the byte
    reference through the gribberish zarr codec and decodes the message.
    This fixture stores a 0..15 ramp, so we can assert the exact values to
    catch a codec that returns garbage rather than just something finite."""
    registry, url = _registry_and_url()
    store = GribberishParser()(url, registry)
    arr = zarr.open_array(store, path=f"{SOIL_GROUP}/stl1", mode="r")

    chunk = np.asarray(arr[0])  # (time0) -> (latitude, longitude)
    np.testing.assert_array_equal(chunk, np.arange(16, dtype=chunk.dtype).reshape(4, 4))
