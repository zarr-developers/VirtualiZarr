"""Translate a configured obstore object into an :class:`icechunk.Storage`.

Used by :meth:`virtualizarr.parsers.icechunk.IcechunkParser.__call__` to bridge
the gap between VirtualiZarr's
[ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] (which holds
obstore-backed stores) and icechunk's own ``Storage`` abstraction (which it
uses to open repositories).

Lives in its own module rather than alongside the parser class because the
dispatch table is its own concern — supporting a new obstore backend means
adding an arm here, not touching the parser. Extracting it also means the
parser module doesn't have to import any of icechunk's storage constructors.

Backend support today: S3, local filesystem, HTTP. PRs for GCS, Azure, R2,
Tigris, etc. welcome — same shape, just another ``isinstance`` arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import icechunk


def obstore_to_icechunk_storage(
    store: Any,
    *,
    relative_prefix: str,
) -> "icechunk.Storage":
    """Build an :class:`icechunk.Storage` from a configured obstore object.

    Handles the common cases (S3, local filesystem, HTTP). Raises a clear
    error for any backend we haven't mapped yet.
    """
    import icechunk
    import obstore.store as obs

    full_prefix = _join_prefix(getattr(store, "prefix", None), relative_prefix)

    if isinstance(store, obs.S3Store):
        cfg = store.config or {}
        client_opts = store.client_options or {}
        # obstore stringifies booleans (e.g. True -> "true"), so we parse the
        # string form rather than treat the value as a Python bool. Also note
        # `allow_http` lives in `client_options`, not `config`.
        skip_signature = str(cfg.get("skip_signature", "")).lower() == "true"
        allow_http = str(client_opts.get("allow_http", "")).lower() == "true"
        return icechunk.s3_storage(
            bucket=cfg["bucket"],
            prefix=full_prefix or None,
            region=cfg.get("region"),
            endpoint_url=cfg.get("endpoint"),
            access_key_id=cfg.get("access_key_id"),
            secret_access_key=cfg.get("secret_access_key"),
            session_token=cfg.get("session_token"),
            anonymous=skip_signature or None,
            allow_http=allow_http,
        )
    if isinstance(store, obs.LocalStore):
        root = Path(store.prefix or "")
        return icechunk.local_filesystem_storage(str(root / relative_prefix))
    if isinstance(store, obs.HTTPStore):
        base = store.url.rstrip("/")
        url = f"{base}/{relative_prefix}" if relative_prefix else base
        return icechunk.http_storage(url)

    raise NotImplementedError(
        f"IcechunkParser doesn't yet know how to translate "
        f"{type(store).__name__} into an icechunk.Storage. "
        f"Either pre-open the icechunk Session yourself and use "
        f"IcechunkParser.parse_session(session, registry), or open an issue."
    )


def _join_prefix(store_prefix: Any, relative: str) -> str:
    """Combine the store's configured prefix with the URL-relative path.

    ``store_prefix`` may be ``None``, a string, or a path-like (obstore's
    ``LocalStore.prefix`` is a ``PosixPath``), so we coerce to ``str`` first.
    """
    left = str(store_prefix or "").strip("/")
    right = relative.strip("/")
    if left and right:
        return f"{left}/{right}"
    return left or right
