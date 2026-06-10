from pathlib import Path

import numpy as np
import pytest
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.tests import requires_grib

# The GribberishParser (and its zarr codec) are new in gribberish 1.0.0.
pytest.importorskip("gribberish", minversion="1.0.0")

from gribberish.virtualizarr import GribberishParser  # noqa: E402

# A 1 KB GRIB1 file with 8 ECMWF soil variables on a 4x4 grid. Vendored from the
# gribberish test suite so this stays a hermetic, offline test.
FIXTURE = Path(__file__).resolve().parents[1] / "data" / "ecmwf_soil_8vars_tiny.grib1"

# 4 soil-temperature layers + 4 soil-moisture layers.
EXPECTED_VARS = {"stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"}


def _registry_and_url() -> tuple[ObjectStoreRegistry, str]:
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return registry, FIXTURE.as_uri()


@requires_grib
def test_grib_virtual_dataset() -> None:
    registry, url = _registry_and_url()
    parser = GribberishParser()
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, Dataset)
        assert set(vds.data_vars) == EXPECTED_VARS
        assert {"latitude", "longitude"}.issubset(set(vds.coords))
        for name in EXPECTED_VARS:
            var = vds[name]
            assert var.dims == ("time", "latitude", "longitude")
            assert var.sizes == {"time": 1, "latitude": 4, "longitude": 4}


@requires_grib
def test_grib_only_variables() -> None:
    registry, url = _registry_and_url()
    parser = GribberishParser(only_variables=["stl1"])
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert set(vds.data_vars) == {"stl1"}


@requires_grib
def test_grib_chunk_decodes_via_codec() -> None:
    """Each GRIB message is one chunk; reading it back resolves the byte
    reference through the gribberish zarr codec and decodes the message."""
    registry, url = _registry_and_url()
    store = GribberishParser()(url, registry)
    arr = zarr.open_array(store, path="stl1", mode="r")

    chunk = np.asarray(arr[0])  # (time0) -> (latitude, longitude)
    assert chunk.shape == (4, 4)
    assert np.isfinite(chunk).all()
