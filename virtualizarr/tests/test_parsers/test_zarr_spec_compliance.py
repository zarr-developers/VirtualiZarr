"""Zarr-spec compliance tests for the ZarrParser — xarray NOT in the loop.

Asserts that virtualizarr's ZarrParser produces ManifestStores whose
metadata matches what zarr-python reads from the same source store.
**xarray is deliberately absent from the assertion path** — the
"reference" is zarr-python itself.

Why this exists alongside `test_zarr_fill_value_equivalence.py`:

The equivalence test compares `virtualizarr+xarray.open_zarr` against
`xarray.open_zarr` direct. When both engines hit xarray's
`FillValueCoder.decode` on a value the coder can't handle (e.g.
`_FillValue=NaN` as a JSON-native number), *both* sides fail with the
same exception — and the test correctly reports it as
`BothEnginesFailedIdenticallyError`. But that "failure" isn't a
virtualizarr bug: the parser is producing zarr-spec-compliant metadata,
and xarray's coder is what can't handle it.

The root cause is that xarray's `_FillValue` encoding uses HDF5-style
base64 even when writing to Zarr (whose metadata is JSON-native). The
fix lives upstream in xarray (tracked at pydata/xarray#11332) — or in
a future zarr-native nodata convention (zarr-specs#351,
zarr-extensions#33).

By testing against zarr-python directly, this module:

1. Asserts virtualizarr correctness independent of xarray's quirks.
2. Stays green even when the equivalence module is red on the same
   spec — surfacing that the gap is upstream, not in virtualizarr.
3. Establishes the "virtualizarr produces zarr-spec-compliant
   manifests" contract as the primary correctness check.

Tests here are deliberately simple: open via parser, open via
zarr-python, compare metadata fields directly. No hypothesis sweeps
(the equivalence module covers the dtype × fill matrix); these are
property-asserting curated cases.

Shared infra (pytestmark, hypothesis profile) is imported from
`_fill_value_common.py` for consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.parsers import ZarrParser
from virtualizarr.tests import requires_arro3
from virtualizarr.tests.test_parsers._fill_value_common import (
    _UNSET,
    base_pytestmark,
    register_hypothesis_profiles,
)

pytestmark = base_pytestmark() + [requires_arro3]
register_hypothesis_profiles()


# ---------------------------------------------------------------------------
# Writer + opener helpers — minimal, since we're not sweeping over a matrix.
# ---------------------------------------------------------------------------


def _write_zarr_simple(
    filepath,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    fill_value: Any = _UNSET,
    attrs: dict[str, Any] | None = None,
) -> str:
    """Write a single-variable Zarr v3 store; return its file:// URL.

    Coordinate arrays are created for each dim so the store is
    well-formed.
    """
    store = zarr.storage.LocalStore(str(filepath))
    root = zarr.group(store=store, zarr_format=3)
    dimension_names = tuple(f"dim_{i}" for i in range(len(shape)))

    arr_kwargs: dict[str, Any] = {
        "name": "data",
        "shape": shape,
        "dtype": dtype,
        "chunks": shape,
        "dimension_names": dimension_names,
    }
    if fill_value is not _UNSET:
        arr_kwargs["fill_value"] = fill_value
    arr = root.create_array(**arr_kwargs)
    arr[...] = np.zeros(shape, dtype=dtype)

    if attrs:
        for k, v in attrs.items():
            arr.attrs[k] = v

    for i, size in enumerate(shape):
        coord = root.create_array(
            name=dimension_names[i],
            shape=(size,),
            dtype="i8",
            chunks=(size,),
            dimension_names=(dimension_names[i],),
        )
        coord[:] = np.arange(size)

    return f"file://{filepath}"


def _open_manifest(url: str):
    """Open the store via virtualizarr's ZarrParser; return a ManifestStore."""
    bare = url.removeprefix("file://")
    store = LocalStore(prefix=bare)
    registry = ObjectStoreRegistry({url: store})
    return ZarrParser()(url=url, registry=registry)


def _open_reference(url: str) -> zarr.Group:
    """Open the store directly with zarr-python."""
    return zarr.open_group(url.removeprefix("file://"), zarr_format=3, mode="r")


def _manifest_data_metadata(ms):
    """Reach into the ManifestStore for the `data` array's metadata.

    `_group` is the (currently underscore-prefixed) accessor used by
    existing tests in `test_zarr.py` and `test_hdf.py`. If/when a
    public accessor lands, update here in one place.
    """
    return ms._group.arrays["data"].metadata


# ---------------------------------------------------------------------------
# Tests — each asserts one spec-compliance property.
# ---------------------------------------------------------------------------


@requires_arro3
class TestZarrSpecCompliance:
    """Virtualizarr's ZarrParser produces metadata that matches what
    zarr-python reads from the same source. No xarray involvement.
    """

    def test_dtype_preserved(self, tmp_path):
        url = _write_zarr_simple(tmp_path / "data.zarr", shape=(3,), dtype="f8")
        ms_meta = _manifest_data_metadata(_open_manifest(url))
        ref_meta = _open_reference(url)["data"].metadata
        assert ms_meta.data_type == ref_meta.data_type

    def test_shape_and_chunks_preserved(self, tmp_path):
        url = _write_zarr_simple(tmp_path / "data.zarr", shape=(4, 6), dtype="i4")
        ms_meta = _manifest_data_metadata(_open_manifest(url))
        ref_meta = _open_reference(url)["data"].metadata
        assert ms_meta.shape == ref_meta.shape
        assert ms_meta.chunk_grid == ref_meta.chunk_grid

    def test_default_fill_value_matches_zarr_python(self, tmp_path):
        """When no explicit fill_value is set, the manifest store's
        `fill_value` should match whatever zarr-python defaults to
        (typically 0 for numeric dtypes).

        Also asserts the original #811 path: uint8 with no explicit
        fill_value doesn't crash and produces sensible metadata.
        """
        url = _write_zarr_simple(tmp_path / "data.zarr", shape=(3,), dtype="u1")
        ms_meta = _manifest_data_metadata(_open_manifest(url))
        ref_meta = _open_reference(url)["data"].metadata
        assert ms_meta.fill_value == ref_meta.fill_value

    def test_nan_fill_value_preserved(self, tmp_path):
        """`fill_value=NaN` on a float dtype round-trips: zarr-python and
        virtualizarr's manifest agree (both store NaN; xarray's
        ability to decode it is a separate question, out of scope here).
        """
        url = _write_zarr_simple(
            tmp_path / "data.zarr",
            shape=(3,),
            dtype="f8",
            fill_value=float("nan"),
        )
        ms_fv = _manifest_data_metadata(_open_manifest(url)).fill_value
        ref_fv = _open_reference(url)["data"].metadata.fill_value
        assert np.isnan(ms_fv) and np.isnan(ref_fv)

    def test_fill_value_attr_is_json_native(self, tmp_path):
        """A `_FillValue` attribute stored as a plain JSON number stays a
        plain JSON number in the manifest — virtualizarr doesn't
        re-encode it into HDF5-style base64 just because xarray's
        `FillValueCoder` would.

        This is the test that distinguishes virtualizarr's contract from
        xarray's: zarr metadata is JSON; scalar attributes should use
        JSON-native types.
        """
        url = _write_zarr_simple(
            tmp_path / "data.zarr",
            shape=(3,),
            dtype="f4",
            attrs={"_FillValue": 0.0},
        )
        attrs = _manifest_data_metadata(_open_manifest(url)).attributes
        assert "_FillValue" in attrs
        assert attrs["_FillValue"] == 0.0
        assert not isinstance(attrs["_FillValue"], (bytes, bytearray, np.ndarray)), (
            f"_FillValue should be a JSON-native scalar, got {type(attrs['_FillValue']).__name__}"
        )

    def test_attrs_match_zarr_python(self, tmp_path):
        """Free-form CF-style attributes (missing_value, valid_range,
        etc.) round-trip with the same JSON shape zarr-python sees.
        """
        url = _write_zarr_simple(
            tmp_path / "data.zarr",
            shape=(3,),
            dtype="f4",
            attrs={
                "_FillValue": 0.0,
                "valid_min": -10.0,
                "valid_max": 10.0,
                "missing_value": -9999.0,
            },
        )
        ms_attrs = dict(_manifest_data_metadata(_open_manifest(url)).attributes)
        ref_attrs = dict(_open_reference(url)["data"].attrs)
        # Filter out any zarr-internal book-keeping keys (none today,
        # but defensive — these tests shouldn't break on future zarr
        # versions adding metadata).
        for k in ("_ARRAY_DIMENSIONS",):
            ms_attrs.pop(k, None)
            ref_attrs.pop(k, None)
        assert ms_attrs == ref_attrs


class TestZarrParserVsXarrayContrast:
    """One-test demonstration of the framing shift: a `_FillValue` value
    that virtualizarr produces spec-compliantly but xarray's
    `FillValueCoder.decode` can't consume.

    Green test here + red test in `test_zarr_fill_value_equivalence.py`
    on the same fixture proves: the gap is upstream of virtualizarr,
    not in the parser.
    """

    def test_nan_fill_value_attr_zarr_compliant_but_xarray_breaks(self, tmp_path):
        """`_FillValue=NaN` as a JSON-native value: virtualizarr's
        ManifestStore preserves it correctly (assertion below passes).
        xarray's `FillValueCoder.decode` on the same store raises
        `TypeError: Failed to decode fill_value: expected str or bytes
        for dtype float64, got float` — see the equivalence module's
        `test_equivalence_curated[float64-nan-fill]` failure. The two
        outcomes together establish the upstream attribution.
        """
        url = _write_zarr_simple(
            tmp_path / "data.zarr",
            shape=(3,),
            dtype="f8",
            fill_value=float("nan"),
            attrs={"_FillValue": float("nan")},
        )
        attrs = _manifest_data_metadata(_open_manifest(url)).attributes
        assert "_FillValue" in attrs
        # Plain Python float, not base64-encoded bytes.
        assert isinstance(attrs["_FillValue"], float)
        assert np.isnan(attrs["_FillValue"])
