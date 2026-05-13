"""Parser for converting an icechunk repository into a VirtualiZarr ManifestStore.

Provides two entry points:

- :meth:`IcechunkParser.__call__(url, registry)` — protocol-conformant. Resolves
  the URL against the registry to find an obstore, translates that obstore into
  an :class:`icechunk.Storage`, opens the repo + a readonly session, and parses.
  Use this when going through :func:`virtualizarr.open_virtual_dataset`.

- :meth:`IcechunkParser.parse_session(session, registry)` — escape hatch. Skip
  the URL/Storage round trip and parse an already-open
  :class:`icechunk.Session` directly. Use this when you already have an open
  Session in hand (the common case if you're working with your icechunk repo
  in the same process).

The mapping is the same regardless of entry point:

- IC virtual ref (any ``s3://`` / ``gs://`` / ``vcc://`` location, already
  resolved by icechunk) → VZ virtual ref with the resolved URL as path.
- IC native (managed) chunk → VZ virtual ref with path
  ``f"{native_chunks_prefix}/{chunk_id}"``.
- IC inline chunk → VZ inline chunk.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from zarr.api.asynchronous import open_group as open_group_async

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.parsers.zarr import metadata_as_v3
from virtualizarr.utils import determine_chunk_grid_shape

if TYPE_CHECKING:
    import icechunk
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry


_DEFAULT_BATCH_SIZE: int = 100_000

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync code, even when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class IcechunkParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an icechunk repository.

    There are two entry points:

    - ``__call__(url, registry)`` matches VirtualiZarr's
      [`Parser`][virtualizarr.parsers.typing.Parser] protocol, so this class
      works with [open_virtual_dataset][virtualizarr.open_virtual_dataset].
      It uses the registry's obstore to identify the icechunk Storage config,
      opens its own ``Repository`` and a readonly ``Session``, and parses.
    - ``parse_session(session, registry)`` is the escape hatch — pass an
      already-open ``icechunk.Session`` and skip the URL/Storage round trip.
      Useful when you already have a session in hand.

    Parameters
    ----------
    native_chunks_prefix
        URL prefix to render icechunk's native (managed) chunk paths under.
        Native chunks become ``f"{native_chunks_prefix}/{chunk_id}"``.

        Optional. When ``__call__(url, ...)`` is used and ``native_chunks_prefix``
        is ``None``, it defaults to ``f"{url}/chunks"`` (which is where icechunk's
        format-constant ``CHUNKS_FILE_PATH`` puts native chunks for a repo at
        that URL). When :meth:`parse_session` is used, this must be supplied
        explicitly — the parser has no URL to derive a default from.
        A single trailing slash is tolerated.
    branch
        Branch name to open in ``__call__`` (default ``"main"``). Ignored by
        ``parse_session`` because the session is already pinned.
    tag
        Tag to open in ``__call__``. Mutually exclusive with ``branch`` /
        ``snapshot_id``.
    snapshot_id
        Snapshot id to open in ``__call__``. Mutually exclusive with
        ``branch`` / ``tag``.
    group
        Optional sub-group path within the icechunk store to use as the root.
    skip_variables
        Names of arrays in the group to exclude.
    batch_size
        Per-batch chunk count for the underlying iterator. Default 100,000.

    Examples
    --------
    >>> import icechunk
    >>> from obspec_utils.registry import ObjectStoreRegistry
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr.parsers import IcechunkParser
    >>>
    >>> # Protocol-conformant path:
    >>> parser = IcechunkParser(  # doctest: +SKIP
    ...     native_chunks_prefix="s3://my-bucket/my-repo/chunks",
    ... )
    >>> vds = open_virtual_dataset(  # doctest: +SKIP
    ...     url="s3://my-bucket/my-repo", registry=registry, parser=parser
    ... )
    >>>
    >>> # Escape hatch — already have a Session:
    >>> repo = icechunk.Repository.open(storage=...)  # doctest: +SKIP
    >>> session = repo.readonly_session(branch="dev")  # doctest: +SKIP
    >>> ms = parser.parse_session(session, registry=registry)  # doctest: +SKIP
    """

    def __init__(
        self,
        native_chunks_prefix: str | None = None,
        *,
        branch: str | None = None,
        tag: str | None = None,
        snapshot_id: str | None = None,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        n_version_specs = sum(v is not None for v in (branch, tag, snapshot_id))
        if n_version_specs > 1:
            raise ValueError(
                "At most one of `branch`, `tag`, `snapshot_id` may be given; "
                f"got branch={branch!r}, tag={tag!r}, snapshot_id={snapshot_id!r}."
            )
        self.branch = branch if n_version_specs else "main"
        self.tag = tag
        self.snapshot_id = snapshot_id

        self.native_chunks_prefix = (
            native_chunks_prefix.rstrip("/")
            if native_chunks_prefix is not None
            else None
        )
        self.group = group
        self.skip_variables = skip_variables
        self.batch_size = batch_size

    def __call__(
        self,
        url: str,
        registry: "ObjectStoreRegistry",
    ) -> ManifestStore:
        """Protocol-conformant entry point: open the icechunk repo from a URL.

        Resolves ``url`` against ``registry`` to find an obstore, translates
        that obstore into an :class:`icechunk.Storage` (currently supports
        S3, local filesystem, and HTTP backends), opens the repository at
        the configured branch/tag/snapshot, and parses.

        If no ``native_chunks_prefix`` was given at construction, it defaults
        to ``f"{url}/chunks"`` — icechunk's format-constant chunks directory
        for the repo at that URL.
        """
        import icechunk

        obstore, relative = registry.resolve(url)
        ic_storage = _obstore_to_icechunk_storage(
            obstore, relative_prefix=str(relative)
        )
        repo = icechunk.Repository.open(storage=ic_storage)
        session = repo.readonly_session(
            branch=self.branch, tag=self.tag, snapshot_id=self.snapshot_id
        )
        prefix = self.native_chunks_prefix or f"{url.rstrip('/')}/chunks"
        return self._parse(session, registry, prefix)

    def parse_session(
        self,
        session: "icechunk.Session",
        registry: "ObjectStoreRegistry",
    ) -> ManifestStore:
        """Escape hatch: parse an already-open icechunk Session directly.

        Bypasses the URL/Storage translation in ``__call__``. The session's
        snapshot is used as-is — the parser's ``branch``/``tag``/``snapshot_id``
        constructor args do not apply on this path.

        ``native_chunks_prefix`` must have been set at construction; without
        a URL, the parser can't derive a default.
        """
        if self.native_chunks_prefix is None:
            raise ValueError(
                "IcechunkParser.parse_session requires native_chunks_prefix "
                "to be set at construction (no URL to derive a default from). "
                "Pass e.g. native_chunks_prefix='s3://my-bucket/my-repo/chunks'."
            )
        return self._parse(session, registry, self.native_chunks_prefix)

    def _parse(
        self,
        session: "icechunk.Session",
        registry: "ObjectStoreRegistry",
        native_chunks_prefix: str,
    ) -> ManifestStore:
        coro = _construct_manifest_group(
            store=session.store,
            group=self.group,
            native_chunks_prefix=native_chunks_prefix,
            skip_variables=self.skip_variables,
            batch_size=self.batch_size,
        )
        manifest_group = _run_async(coro)
        return ManifestStore(registry=registry, group=manifest_group)


def _obstore_to_icechunk_storage(
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
        return icechunk.s3_storage(
            bucket=cfg["bucket"],
            prefix=full_prefix or None,
            region=cfg.get("region"),
            endpoint_url=cfg.get("endpoint"),
            access_key_id=cfg.get("access_key_id"),
            secret_access_key=cfg.get("secret_access_key"),
            session_token=cfg.get("session_token"),
            anonymous=cfg.get("skip_signature", False) or None,
            allow_http=cfg.get("allow_http", False),
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


async def _construct_manifest_group(
    store: "icechunk.IcechunkStore",
    *,
    native_chunks_prefix: str,
    group: str | None = None,
    skip_variables: Iterable[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ManifestGroup:
    """Build a ManifestGroup from an icechunk zarr group."""
    zarr_group = await open_group_async(store=store, path=group, mode="r")

    array_keys = [key async for key in zarr_group.array_keys()]
    skip = set() if skip_variables is None else set(skip_variables)

    zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(k) for k in array_keys if k not in skip]
    )

    manifest_arrays = await asyncio.gather(
        *[
            _construct_manifest_array(arr, store, native_chunks_prefix, batch_size)
            for arr in zarr_arrays  # type: ignore[union-attr]
        ]
    )

    arrays = {a.basename: ma for a, ma in zip(zarr_arrays, manifest_arrays)}
    return ManifestGroup(arrays=arrays, attributes=dict(zarr_group.attrs))


async def _construct_manifest_array(
    zarr_array: "zarr.AsyncArray[Any]",
    store: "icechunk.IcechunkStore",
    native_chunks_prefix: str,
    batch_size: int,
) -> ManifestArray:
    """Assemble one array's ChunkManifest by scattering iterator batches."""
    # Import ChunkType lazily so importing this module doesn't require
    # icechunk at install time (matches the TYPE_CHECKING-only import above).
    from icechunk import ChunkType

    metadata = metadata_as_v3(zarr_array.metadata)
    grid_shape = determine_chunk_grid_shape(
        metadata.shape, metadata.chunk_grid.chunk_shape
    )

    total = int(np.prod(grid_shape)) if grid_shape else 1
    paths = np.full(total, "", dtype=np.dtypes.StringDType())
    offsets = np.zeros(total, dtype=np.uint64)
    lengths = np.zeros(total, dtype=np.uint64)
    inlined: dict[tuple[int, ...], bytes] = {}

    prefix_with_slash = f"{native_chunks_prefix}/"

    async for batch in store.array_chunk_iterator(zarr_array.path, batch_size):
        b_coords, b_kinds, b_paths, b_offsets, b_lengths, b_inlined = batch
        n = b_coords.shape[0]
        if n == 0:
            continue

        if grid_shape:
            flat_idx = np.ravel_multi_index(b_coords.T, grid_shape)
        else:
            flat_idx = np.zeros(n, dtype=np.intp)

        path_col = np.array(b_paths, dtype=np.dtypes.StringDType())
        is_native = b_kinds == ChunkType.native
        if is_native.any():
            path_col[is_native] = np.strings.add(prefix_with_slash, path_col[is_native])
        is_inline = b_kinds == ChunkType.inline
        if is_inline.any():
            path_col[is_inline] = INLINED_CHUNK_PATH

        paths[flat_idx] = path_col
        offsets[flat_idx] = b_offsets
        lengths[flat_idx] = b_lengths

        for batch_i, data in b_inlined.items():
            coord = tuple(int(x) for x in b_coords[batch_i])
            inlined[coord] = data

    chunk_manifest = ChunkManifest.from_arrays(
        paths=paths.reshape(grid_shape or (1,)),
        offsets=offsets.reshape(grid_shape or (1,)),
        lengths=lengths.reshape(grid_shape or (1,)),
        validate_paths=False,
        inlined=inlined or None,
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
