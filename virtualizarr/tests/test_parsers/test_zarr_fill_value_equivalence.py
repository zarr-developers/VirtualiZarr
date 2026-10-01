"""Property-based equivalence tests for the ZarrParser's fill-value handling.

Each test writes a Zarr v3 store via zarr-python, then opens it two ways:

  1. `xr.open_zarr(path)` directly (reference) — the user's "no virtualizarr"
     baseline.
  2. virtualizarr's `ZarrParser` → `ManifestStore` → `xr.open_dataset(...,
     engine="zarr")` (observed) — the under-test path.

Both go through xarray's zarr backend ultimately; we're asserting that
virtualizarr's manifest-construction layer is transparent. Failures
indicate the parser is either dropping/transforming metadata or
mis-resolving fill values during manifest construction.

Test philosophy mirrors the HDF equivalence module: strategies generate
the full parameter range that should work; assertions are plain (no
`xfail`, no `pytest.raises`). Failures are TODO items.

Concrete issue coverage:

- **#811** (`ZarrParser get_metadata fails for LocalStore Zarr with
  dtype uint8`): a curated example with `dtype=uint8` and no explicit
  fill_value exercises the default-fill-lookup path that crashed in
  the bug report.
- **Cross-parser fill-value consistency:** the random strategy
  generates every numeric dtype with various fill-value shapes,
  surfacing any divergence from xarray's direct view.

Shared infrastructure (sentinel, hypothesis profiles, equivalence
helpers, generic dtype strategies) lives in
`virtualizarr/tests/test_parsers/_fill_value_common.py`.

CI throttling:
- VIRTUALIZARR_HYPOTHESIS_PROFILE=ci      -> max_examples=10  (faster)
- VIRTUALIZARR_HYPOTHESIS_PROFILE=thorough -> max_examples=50  (default)
- pytest -m "not hypothesis_tests"        -> skip this module entirely
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.parsers import ZarrParser
from virtualizarr.tests import requires_arro3
from virtualizarr.tests.test_parsers._fill_value_common import (
    _UNSET,
    assert_equivalent,
    base_numeric_dtype_strategy,
    base_pytestmark,
    data_in_dtype_strategy,
    register_hypothesis_profiles,
    value_in_dtype_strategy,
)

pytestmark = base_pytestmark() + [requires_arro3]
register_hypothesis_profiles()


# ---------------------------------------------------------------------------
# _DatasetSpec — Zarr-flavoured. Compared to the HDF one this is much
# simpler: no h5py `dataset.fillvalue` / explicit-vs-default distinction
# (zarr v3 always has a fill_value), no _Unsigned, no scale_factor/add_offset
# in the metadata (those would be plain attributes if present).
# ---------------------------------------------------------------------------


@dataclass
class _DatasetSpec:
    """A self-contained description of one Zarr v3 store to write for testing.

    Fields:
        dtype: the array's numeric dtype.
        data: the data to write.
        fill_value: zarr's storage-level `fill_value`. `_UNSET` means "let
            zarr-python pick the dtype default" (which is what triggers
            #811 for uint8).
        fill_value_attr: the CF `_FillValue` attribute on the array. Set
            independently of `fill_value` — they're allowed to disagree
            in CF.
        chunks: explicit chunk shape; None means single-chunk.
        extra_attrs: free-form CF-style attributes to round-trip
            (`missing_value`, `valid_range`, etc.).
    """

    dtype: np.dtype
    data: np.ndarray
    fill_value: Any = _UNSET
    fill_value_attr: Any = _UNSET
    chunks: tuple[int, ...] | None = None
    extra_attrs: dict[str, Any] | None = None


def _write_zarr(
    filepath: Path | str,
    *,
    dtype: np.dtype,
    data: np.ndarray,
    fill_value: Any = _UNSET,
    fill_value_attr: Any = _UNSET,
    chunks: tuple[int, ...] | None = None,
    extra_attrs: dict[str, Any] | None = None,
) -> str:
    """Write a single-variable Zarr v3 store with the requested shape.

    Returns the file:// URL for use with the ZarrParser. Coordinate
    variables (one per dim) are written too so xarray's zarr backend
    builds a proper Dataset with named dims rather than anonymous ones.
    """
    filepath = str(filepath)
    store = zarr.storage.LocalStore(filepath)
    root = zarr.group(store=store, zarr_format=3)

    dimension_names = tuple(f"dim_{i}" for i in range(data.ndim))
    chunk_shape = chunks if chunks is not None else data.shape

    # zarr-python's `create_array` requires a fill_value. Passing
    # `fill_value=None` would explicitly write null; to exercise the
    # "user didn't specify" code path (issue #811) we want to omit the
    # kwarg entirely.
    arr_kwargs: dict[str, Any] = {
        "name": "data",
        "shape": data.shape,
        "dtype": dtype,
        "chunks": chunk_shape,
        "dimension_names": dimension_names,
    }
    if fill_value is not _UNSET:
        arr_kwargs["fill_value"] = fill_value
    arr = root.create_array(**arr_kwargs)
    arr[...] = data

    # zarr v3 attributes are stored in JSON, which doesn't accept numpy
    # scalars. Real users storing CF-style fill values write Python
    # primitives (int / float / str / list), so coerce here — this isn't
    # a fidelity loss for the equivalence test, just matches the user
    # contract. (How virtualizarr's parsers handle numpy-scalar attrs
    # they read from h5py / etc. is a separate concern — see Issue #715.)
    def _to_json_safe(v: Any) -> Any:
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    if fill_value_attr is not _UNSET:
        arr.attrs["_FillValue"] = _to_json_safe(fill_value_attr)
    if extra_attrs is not None:
        for k, v in extra_attrs.items():
            arr.attrs[k] = _to_json_safe(v)

    # Write coordinate arrays for each dim so xarray sees a proper
    # named-dim Dataset rather than just a bare DataArray.
    for axis, (dim_name, size) in enumerate(zip(dimension_names, data.shape)):
        coord = root.create_array(
            name=dim_name,
            shape=(size,),
            dtype="i8",
            chunks=(size,),
            dimension_names=(dim_name,),
        )
        coord[:] = np.arange(size)

    return f"file://{filepath}"


# ---------------------------------------------------------------------------
# Open callbacks: virtualizarr-via-ZarrParser (observed) vs xarray-direct
# (reference). Both ultimately go through xarray's zarr backend; the
# difference is whether virtualizarr's manifest layer sits in between.
# ---------------------------------------------------------------------------


def _open_virtualizarr(filepath: str, *, decode_cf: bool) -> xr.Dataset:
    """Open `filepath` via virtualizarr's ZarrParser. Plugged into the
    shared equivalence helpers as the `open_observed` callback.
    """
    url = f"file://{filepath}"
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({url: store})
    parser = ZarrParser()
    manifest_store = parser(url=url, registry=registry)
    return xr.open_dataset(
        manifest_store,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        decode_cf=decode_cf,
    )


def _open_zarr_direct(filepath: str, *, decode_cf: bool) -> xr.Dataset:
    """Open `filepath` directly via `xr.open_zarr` — no virtualizarr. The
    reference for the equivalence sweep.
    """
    return xr.open_dataset(
        filepath,
        engine="zarr",
        consolidated=False,
        zarr_format=3,
        decode_cf=decode_cf,
    )


def _assert_zarrparser_xarray_identical(filepath: str) -> None:
    """Run both raw-attribute and decoded-data invariants against the Zarr
    parser. Thin wrapper that binds the engines.
    """
    assert_equivalent(
        filepath,
        open_observed=_open_virtualizarr,
        open_reference=_open_zarr_direct,
    )


# ---------------------------------------------------------------------------
# Sanity tests — confirm the writer and equivalence helper work on a
# trivially-trivial file before any hypothesis runs.
# ---------------------------------------------------------------------------


def test_write_zarr_basic_roundtrip(tmp_path):
    """`_write_zarr` produces a Zarr store with the expected shape, dtype,
    and fill_value that zarr-python can read back."""
    url = _write_zarr(
        tmp_path / "data.zarr",
        dtype=np.dtype("f8"),
        data=np.array([1.0, 2.0, 3.0], dtype="f8"),
        fill_value=-9999.0,
        chunks=(2,),
    )
    assert url.startswith("file://")

    arr = zarr.open_array(
        store=zarr.storage.LocalStore(url.removeprefix("file://")),
        path="data",
        mode="r",
    )
    np.testing.assert_array_equal(arr[:], [1.0, 2.0, 3.0])
    assert arr.metadata.fill_value == -9999.0


def test_equivalence_helper_passes_on_simple_zarr(tmp_path):
    """A trivial float64 store should round-trip identically through both
    open paths."""
    url = _write_zarr(
        tmp_path / "simple.zarr",
        dtype=np.dtype("f8"),
        data=np.array([1.0, 2.0, 3.0, 4.0], dtype="f8"),
        fill_value=0.0,
    )
    _assert_zarrparser_xarray_identical(url.removeprefix("file://"))


# ---------------------------------------------------------------------------
# Random-draw strategy and basic equivalence sweep.
#
# Strategy generates the full numeric dtype matrix from
# base_numeric_dtype_strategy() (bool / signed int / unsigned int / float /
# complex) plus several shapes / fill-value patterns. Failures from
# random draws are real signals to investigate; the curated `@example`s
# pin specific bug-report cases (#811).
# ---------------------------------------------------------------------------


@st.composite
def _basic_zarr_dataset_strategy(draw) -> _DatasetSpec:
    dtype = draw(base_numeric_dtype_strategy())
    shape = draw(npst.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=8))
    chunks_choice = draw(st.booleans())
    chunks = (
        tuple(draw(st.integers(min_value=1, max_value=size)) for size in shape)
        if chunks_choice
        else None
    )
    data = draw(data_in_dtype_strategy(dtype, shape))
    fill_in_dtype = draw(value_in_dtype_strategy(dtype))
    # zarr v3 requires fill_value to be of the array dtype (or None for
    # numeric "default zero"). We sample three options: omit entirely
    # (triggers default-fill path — #811), set to a typed value, or set
    # to a typed value and *also* expose it as a CF _FillValue attr.
    fill_value = draw(st.sampled_from([_UNSET, fill_in_dtype]))
    fill_value_attr = draw(st.sampled_from([_UNSET, fill_in_dtype]))
    return _DatasetSpec(
        dtype=dtype,
        data=data,
        fill_value=fill_value,
        fill_value_attr=fill_value_attr,
        chunks=chunks,
    )


def _spec_to_url(spec: _DatasetSpec, tmp_path: Path) -> str:
    """Translate a `_DatasetSpec` to a written Zarr store and return its
    bare path (no file:// prefix), for use with the equivalence helper.
    """
    url = _write_zarr(
        tmp_path / "data.zarr",
        dtype=spec.dtype,
        data=spec.data,
        fill_value=spec.fill_value,
        fill_value_attr=spec.fill_value_attr,
        chunks=spec.chunks,
        extra_attrs=spec.extra_attrs,
    )
    return url.removeprefix("file://")


# Curated examples. Each pins a specific bug-report case or scenario we
# want guaranteed coverage of, separate from the random draws.
_BASIC_EXAMPLES = [
    # Issue #811: uint8 with no explicit fill_value — exercises the
    # default-fill-lookup path that crashed `get_metadata` in the bug
    # report. The Zarr v3 default for uint8 is 0; both engines should
    # agree.
    _DatasetSpec(
        dtype=np.dtype("u1"),
        data=np.array([1, 2, 3], dtype="u1"),
    ),
    # uint8 *with* an explicit fill_value — same dtype, but bypassing
    # the default-lookup path. If only the default-lookup branch is
    # broken, this passes while the no-fill case fails — useful signal.
    _DatasetSpec(
        dtype=np.dtype("u1"),
        data=np.array([1, 2, 3], dtype="u1"),
        fill_value=np.uint8(255),
    ),
    # NaN-valued _FillValue on float64. Tests NaN equality semantics
    # through the parser.
    _DatasetSpec(
        dtype=np.dtype("f8"),
        data=np.array([1.0, 2.0, 3.0], dtype="f8"),
        fill_value=np.nan,
        fill_value_attr=np.nan,
    ),
    # int32 with both fill_value and matching CF _FillValue.
    _DatasetSpec(
        dtype=np.dtype("i4"),
        data=np.array([10, 20, 30], dtype="i4"),
        fill_value=np.int32(-9999),
        fill_value_attr=np.int32(-9999),
    ),
    # bool array with explicit fill_value=True.
    _DatasetSpec(
        dtype=np.dtype("?"),
        data=np.array([True, False, True], dtype="?"),
        fill_value=True,
    ),
]


_BASIC_EXAMPLE_IDS = [
    "uint8-no-fill",  # #811 reproducer
    "uint8-with-fill",
    "float64-nan-fill",
    "int32-matching-fill",
    "bool-fill",
]


def _ensure_clean_zarr_path(tmp_path: Path) -> None:
    """Zarr's LocalStore needs an empty directory each draw. Tear down
    any previous `data.zarr` so the writer starts clean.
    """
    store_path = tmp_path / "data.zarr"
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)


@requires_arro3
class TestBasicEquivalence:
    """Equivalence sweep across the Zarr numeric dtype × fill matrix.

    Curated examples and random draws are split into separate test
    methods so pytest-xdist can parallelise the curated cases and a
    single failing example doesn't shadow the others.

    Assertion is plain `_assert_zarrparser_xarray_identical`. Cases the
    parser/stack doesn't handle correctly today **will fail** — those
    failures are the to-fix list.
    """

    @pytest.mark.parametrize("spec", _BASIC_EXAMPLES, ids=_BASIC_EXAMPLE_IDS)
    def test_equivalence_curated(self, spec, tmp_path):
        _ensure_clean_zarr_path(tmp_path)
        filepath = _spec_to_url(spec, tmp_path)
        _assert_zarrparser_xarray_identical(filepath)

    # tmp_path is function-scoped, but each hypothesis example writes the
    # same `data.zarr` and reads it before the next example runs. The
    # overwrite-in-place is intentional.
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(spec=_basic_zarr_dataset_strategy())
    def test_equivalence_random(self, spec, tmp_path):
        _ensure_clean_zarr_path(tmp_path)
        filepath = _spec_to_url(spec, tmp_path)
        _assert_zarrparser_xarray_identical(filepath)
