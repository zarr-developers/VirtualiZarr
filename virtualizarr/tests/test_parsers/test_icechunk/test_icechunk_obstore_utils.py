"""Unit tests for the obstore -> icechunk.Storage translation helper.

The translator is internal to ``IcechunkParser.__call__`` but exercised
separately here so failures point at the dispatch table rather than at the
end-to-end parse path.
"""

from __future__ import annotations

import pytest

icechunk = pytest.importorskip("icechunk")
obstore_store = pytest.importorskip("obstore.store")

from virtualizarr.parsers.icechunk.obstore_utils import obstore_to_icechunk_storage


def test_s3_bool_config_translated_correctly() -> None:
    """Obstore stringifies bool config (e.g. ``True`` becomes ``"true"``).

    The translation must parse the string form, not pass it through —
    otherwise ``s3_storage(anonymous="true", ...)`` raises a TypeError because
    the icechunk arg expects ``bool | None``. ``allow_http`` lives in
    ``store.client_options``, not ``store.config`` — also covered here.

    See https://github.com/zarr-developers/VirtualiZarr/pull/991 (tylanderson).
    """
    store = obstore_store.S3Store(
        bucket="mybucket",
        config={"skip_signature": True},
        client_options={"allow_http": True},
    )
    # Confirm the preconditions that motivated this test: obstore stringifies
    # bools, and allow_http lives on client_options not config.
    assert store.config["skip_signature"] == "true"
    assert store.client_options["allow_http"] == "true"

    # Translation must produce a valid icechunk.Storage — if either bool
    # leaked through as a string, s3_storage would raise a TypeError here.
    storage = obstore_to_icechunk_storage(store, relative_prefix="repo")
    assert isinstance(storage, icechunk.Storage)
