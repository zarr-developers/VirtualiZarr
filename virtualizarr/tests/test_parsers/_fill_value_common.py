"""Shared infrastructure for parser fill-value equivalence test modules.

Each parser has its own equivalence test module (`test_hdf_fill_value_equivalence.py`,
`test_zarr_fill_value_equivalence.py`, ...). The format-agnostic pieces live
here so adding parser N+1 means writing only the parser-specific bits:

- a `_write_<format>` file writer
- an `open_observed` and `open_reference` pair of callables
- format-specific dtype / fill-value strategies
- a `_DatasetSpec`-shaped dataclass with any format-specific extra fields
- the test classes themselves

The shared infrastructure here covers: the `_UNSET` sentinel, hypothesis
profile registration, the module-level `pytestmark` items, two-layer
equivalence assertion helpers, and a generic numeric-dtype strategy.

Test philosophy: strategies generate the full parameter range
`docs/custom_parsers.md` says the parser should support. Assertions are
plain `assert_identical` — no `xfail`, no `pytest.raises` for
known-broken combinations. Failures are TODO items. Suite green =
parser stack matches spec.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

# ---------------------------------------------------------------------------
# Sentinel for "this parameter was not set" — `None` is a meaningful value
# in several places (e.g. zarr's `fill_value=None` means "use default"), so
# we need a third state.
# ---------------------------------------------------------------------------


class _Unset:
    """Sentinel for parameters that should be omitted entirely.

    None is a meaningful value for most fill/encoding knobs, so we need a
    third state. Used as the default value on parser-specific
    `_DatasetSpec` dataclasses so the test writer can pass `_UNSET` to
    mean "don't set this attribute at all".
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):  # pragma: no cover
        return "_UNSET"


_UNSET = _Unset()


# ---------------------------------------------------------------------------
# Hypothesis profile registration and the module-level pytestmark items
# that every fill-value equivalence module should set.
# ---------------------------------------------------------------------------


def register_hypothesis_profiles() -> None:
    """Register `ci` (max_examples=10) and `thorough` (max_examples=50)
    profiles and load whichever `VIRTUALIZARR_HYPOTHESIS_PROFILE` names
    (default: `thorough`).

    Idempotent — safe to call multiple times. Each fill-value equivalence
    test module calls this at import time. `register_profile` overwrites
    existing profiles of the same name, so re-calling is harmless.
    """
    settings.register_profile("ci", deadline=None, max_examples=10)
    settings.register_profile("thorough", deadline=None, max_examples=50)
    settings.load_profile(os.environ.get("VIRTUALIZARR_HYPOTHESIS_PROFILE", "thorough"))


def base_pytestmark() -> list:
    """Return the module-level pytestmark list every fill-value equivalence
    module uses: the `hypothesis_tests` marker plus a filter for zarr's
    "this dtype has no Zarr V3 spec" warnings.

    Returns a fresh list each call so callers can extend it without
    mutating shared state.
    """
    return [
        pytest.mark.hypothesis_tests,
        # Several test cases deliberately exercise dtypes like S* that
        # don't yet have a Zarr V3 spec; zarr-python emits an
        # informational `UnstableSpecificationWarning` on every read. We
        # want those dtypes covered, so silence the warning module-locally.
        pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
    ]


# ---------------------------------------------------------------------------
# Two-layer equivalence assertion helpers.
#
# Each parser test module provides `open_observed` and `open_reference`
# callbacks that take (filepath, *, decode_cf) and return an xr.Dataset.
# The helpers below run both opens, clear the legitimately-different
# `encoding` dicts, and assert dataset identity.
#
# Attribution: when one or both opens fail, the helpers produce specific
# error messages so the developer can tell whether the failure is
# virtualizarr-specific or downstream-shared.
# ---------------------------------------------------------------------------


OpenCallable = Callable[..., xr.Dataset]


class BothEnginesFailedIdenticallyError(AssertionError):
    """Both engines raised matching exceptions when opening the test file.

    Still a spec failure (the parser stack doesn't satisfy what
    `docs/custom_parsers.md` says should work), but **not a
    virtualizarr-specific bug** — the fix likely lives downstream
    (xarray's `FillValueCoder`, zarr-python, etc.). The exception
    message includes the signature so the developer can attribute the
    fix correctly.
    """


def _exception_signature(e: BaseException) -> str:
    """Stable string signature for comparing exceptions.

    Truncates the message to avoid noise from path differences and
    long tracebacks. Two exceptions are considered "matching" when
    their signatures are equal.
    """
    msg = str(e)
    # Trim — many xarray/zarr errors include long type repr lines.
    if len(msg) > 200:
        msg = msg[:200]
    return f"{type(e).__name__}: {msg}"


_OPEN_OK = "ok"
_OPEN_FAIL = "fail"


def _open_with_capture(
    open_fn: OpenCallable, filepath: str, *, decode_cf: bool
) -> tuple[str, Any]:
    """Try `open_fn(filepath, decode_cf=decode_cf)`; return either
    (`_OPEN_OK`, dataset) or (`_OPEN_FAIL`, exception).
    """
    try:
        return (_OPEN_OK, open_fn(filepath, decode_cf=decode_cf))
    except BaseException as e:  # pragma: no cover — exhaustive catch
        return (_OPEN_FAIL, e)


def _assert_pair_identical(
    filepath: str,
    *,
    decode_cf: bool,
    layer_name: str,
    open_observed: OpenCallable,
    open_reference: OpenCallable,
) -> None:
    """Open both engines at the given decode layer and assert they agree.

    Four outcome shapes, each producing a distinct failure mode:

    - both succeed → standard `assert_identical` comparison.
    - both fail with matching exception signatures →
      `BothEnginesFailedIdenticallyError` (downstream-shared bug).
    - both fail differently → plain `AssertionError` with both sigs.
    - one fails, one succeeds → `AssertionError` naming which side
      failed.
    """
    obs_status, obs = _open_with_capture(open_observed, filepath, decode_cf=decode_cf)
    ref_status, ref = _open_with_capture(open_reference, filepath, decode_cf=decode_cf)

    # Case: both failed.
    if obs_status == _OPEN_FAIL and ref_status == _OPEN_FAIL:
        obs_sig = _exception_signature(obs)
        ref_sig = _exception_signature(ref)
        if obs_sig == ref_sig:
            raise BothEnginesFailedIdenticallyError(
                f"[{layer_name}] both engines failed identically — "
                f"likely a downstream-shared issue, not a virtualizarr-"
                f"specific bug.\n  {obs_sig}"
            ) from obs
        raise AssertionError(
            f"[{layer_name}] both engines failed but differently — "
            f"unexpected divergence.\n"
            f"  observed (virtualizarr): {obs_sig}\n"
            f"  reference:               {ref_sig}"
        ) from obs

    # Case: only one failed.
    if obs_status == _OPEN_FAIL:
        ref.close()
        raise AssertionError(
            f"[{layer_name}] observed (virtualizarr) failed; reference "
            f"succeeded — likely a virtualizarr-specific bug.\n"
            f"  observed: {_exception_signature(obs)}"
        ) from obs
    if ref_status == _OPEN_FAIL:
        obs.close()
        raise AssertionError(
            f"[{layer_name}] reference failed; observed (virtualizarr) "
            f"succeeded — unexpected (virtualizarr accepts what the "
            f"reference engine doesn't).\n"
            f"  reference: {_exception_signature(ref)}"
        ) from ref

    # Case: both succeeded — compare.
    try:
        # Encoding dicts legitimately differ between engines (HDF5 chunk
        # / filter encoding vs Zarr codec encoding), so clear them
        # before comparing. Attribute and value comparison stays strict.
        for v in obs.variables:
            obs[v].encoding.clear()
        for v in ref.variables:
            ref[v].encoding.clear()
        xr.testing.assert_identical(obs.load(), ref.load())
    finally:
        obs.close()
        ref.close()


def assert_decoded_data_identical(
    filepath: str,
    *,
    open_observed: OpenCallable,
    open_reference: OpenCallable,
) -> None:
    """Assert the two engines produce identical datasets *after* xarray's
    CF decoding (mask_and_scale, _FillValue masking, scale_factor/add_offset,
    _Unsigned reinterpretation).

    User-facing correctness check: does the data look the same when
    consumed via either engine?
    """
    _assert_pair_identical(
        filepath,
        decode_cf=True,
        layer_name="decoded-data",
        open_observed=open_observed,
        open_reference=open_reference,
    )


def assert_raw_attributes_identical(
    filepath: str,
    *,
    open_observed: OpenCallable,
    open_reference: OpenCallable,
) -> None:
    """Assert the two engines produce identical raw datasets with CF decoding
    *off*.

    Exposes the metadata layer directly: any attribute the parser drops,
    adds, or transforms (e.g. `_FillValue` encoding mismatches, lost
    `_Unsigned`, missing `missing_value`) shows up here even though the
    decoded-data helper would mask it via CF decoding.
    """
    _assert_pair_identical(
        filepath,
        decode_cf=False,
        layer_name="raw-attributes",
        open_observed=open_observed,
        open_reference=open_reference,
    )


def assert_equivalent(
    filepath: str,
    *,
    open_observed: OpenCallable,
    open_reference: OpenCallable,
) -> None:
    """Convenience wrapper: run both raw-attribute and decoded-data assertions."""
    assert_raw_attributes_identical(
        filepath, open_observed=open_observed, open_reference=open_reference
    )
    assert_decoded_data_identical(
        filepath, open_observed=open_observed, open_reference=open_reference
    )


# ---------------------------------------------------------------------------
# Generic numeric-dtype strategies shared across parser modules.
#
# Each parser module adds format-specific extensions (HDF adds vlen-string
# and S*, TIFF restricts to GDAL-supported dtypes, etc.).
# ---------------------------------------------------------------------------


def base_numeric_dtype_strategy() -> st.SearchStrategy[np.dtype]:
    """Sample one of the numeric dtypes every format supports: bool,
    signed/unsigned int, float, complex. Excludes string and structured
    dtypes — those are too format-specific to share.
    """
    return st.sampled_from(
        [
            np.dtype("?"),  # bool
            np.dtype("i1"),
            np.dtype("i2"),
            np.dtype("i4"),
            np.dtype("i8"),
            np.dtype("u1"),
            np.dtype("u2"),
            np.dtype("u4"),
            np.dtype("u8"),
            np.dtype("f4"),
            np.dtype("f8"),
            np.dtype("c8"),
            np.dtype("c16"),
        ]
    )


def value_in_dtype_strategy(dtype: np.dtype) -> st.SearchStrategy[Any]:
    """Return a strategy producing a value compatible with `dtype`.

    Used for both `dataset_fillvalue` and `_FillValue` draws. Doing the
    dtype-aware draw at strategy-build time (not via `assume()`) keeps
    shrinking efficient.

    Handles the universally-supported kinds (`b`, `i`, `u`, `f`, `c`, `S`).
    Parser modules that exercise additional kinds (e.g. `O` for
    h5py.string_dtype()) extend this in their own helper.
    """
    if dtype.kind == "b":
        return st.booleans()
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return st.integers(min_value=int(info.min), max_value=int(info.max))
    if dtype.kind == "f":
        return st.floats(
            width=64 if dtype.itemsize == 8 else 32,
            allow_nan=True,
            allow_infinity=False,
        )
    if dtype.kind == "c":
        component = st.floats(
            width=32 if dtype.itemsize == 8 else 64,
            allow_nan=False,
            allow_infinity=False,
        )
        return st.builds(complex, component, component)
    if dtype.kind == "S":
        return st.binary(min_size=0, max_size=dtype.itemsize)
    raise ValueError(
        f"No fill-value strategy for dtype {dtype!r}; extend the parser "
        f"module's own strategy if you need this dtype."
    )


def data_in_dtype_strategy(
    dtype: np.dtype, shape: tuple[int, ...]
) -> st.SearchStrategy[np.ndarray]:
    """Generate a numpy array of the given dtype and shape for the dtypes
    handled by `value_in_dtype_strategy`.

    Parser modules whose strategies introduce additional kinds (e.g. vlen
    strings) should branch in their own data-strategy helper.
    """
    return npst.arrays(dtype=dtype, shape=shape)
