"""IcechunkParser: walks an icechunk repository and builds a VZ ManifestStore.

The user-facing class is :class:`IcechunkParser`. The obstore →
:class:`icechunk.Storage` translation it uses on the URL path lives in the
sibling :mod:`.obstore_utils` module.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine, Iterable
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
from virtualizarr.parsers.icechunk.obstore_utils import obstore_to_icechunk_storage
from virtualizarr.parsers.zarr import metadata_as_v3
from virtualizarr.utils import determine_chunk_grid_shape

if TYPE_CHECKING:
    import icechunk
    import zarr
    from obspec_utils.registry import ObjectStoreRegistry


_DEFAULT_BATCH_SIZE: int = 100_000

T = TypeVar("T")


def _require_icechunk_with_array_chunk_iterator() -> None:
    """Check the installed icechunk has the API IcechunkParser needs.

    ``IcechunkStore.array_chunk_iterator`` was added in icechunk 2.0.5. We
    don't bump VirtualiZarr's ``[icechunk]`` extra to ``>=2.0.5`` because
    most users want only the writer (``to_icechunk``), which works against
    older icechunk versions — paying the parser's higher floor would force
    them to upgrade for no benefit. Instead we feature-detect here and fail
    fast with a clear message when someone constructs the parser against an
    older icechunk.
    """
    try:
        from icechunk import IcechunkStore
    except ImportError as e:
        raise ImportError(
            "IcechunkParser requires icechunk. Install with "
            "`pip install icechunk>=2.0.5` (or "
            "`pip install virtualizarr[icechunk]` then upgrade icechunk)."
        ) from e
    if not hasattr(IcechunkStore, "array_chunk_iterator"):
        import icechunk

        raise ImportError(
            f"IcechunkParser requires icechunk>=2.0.5 (for the new "
            f"IcechunkStore.array_chunk_iterator API), but found "
            f"icechunk=={icechunk.__version__}. Upgrade with "
            f"`pip install --upgrade icechunk`."
        )


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
    >>> import icechunk  # doctest: +SKIP
    >>> from virtualizarr import open_virtual_dataset
    >>> from virtualizarr.parsers import IcechunkParser
    >>>
    >>> # Protocol-conformant path — native chunks rendered as
    >>> # ``f"{url}/chunks/{id}"`` automatically.
    >>> vds = open_virtual_dataset(  # doctest: +SKIP
    ...     url="s3://my-bucket/my-repo",
    ...     registry=registry,
    ...     parser=IcechunkParser(),
    ... )
    >>>
    >>> # Escape hatch — already have a Session. Native-chunks prefix must be
    >>> # supplied here since there's no URL.
    >>> repo = icechunk.Repository.open(storage=...)  # doctest: +SKIP
    >>> session = repo.readonly_session(branch="dev")  # doctest: +SKIP
    >>> ms = IcechunkParser().parse_session(  # doctest: +SKIP
    ...     session,
    ...     registry=registry,
    ...     native_chunks_prefix="s3://my-bucket/my-repo/chunks",
    ... )
    """

    def __init__(
        self,
        *,
        branch: str | None = None,
        tag: str | None = None,
        snapshot_id: str | None = None,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        _require_icechunk_with_array_chunk_iterator()

        n_version_specs = sum(v is not None for v in (branch, tag, snapshot_id))
        if n_version_specs > 1:
            raise ValueError(
                "At most one of `branch`, `tag`, `snapshot_id` may be given; "
                f"got branch={branch!r}, tag={tag!r}, snapshot_id={snapshot_id!r}."
            )
        self.branch = branch if n_version_specs else "main"
        self.tag = tag
        self.snapshot_id = snapshot_id

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
        the configured branch/tag/snapshot, and parses. Native chunk paths
        are rendered as ``f"{url}/chunks/{chunk_id}"`` — icechunk's
        format-constant chunks directory for the repo at that URL.
        """
        import icechunk

        obstore, relative = registry.resolve(url)
        ic_storage = obstore_to_icechunk_storage(obstore, relative_prefix=str(relative))
        repo = icechunk.Repository.open(storage=ic_storage)
        session = repo.readonly_session(
            branch=self.branch, tag=self.tag, snapshot_id=self.snapshot_id
        )
        return self.parse_session(
            session,
            registry,
            native_chunks_prefix=f"{url.rstrip('/')}/chunks",
        )

    def parse_session(
        self,
        session: "icechunk.Session",
        registry: "ObjectStoreRegistry",
        *,
        native_chunks_prefix: str,
    ) -> ManifestStore:
        """Escape hatch: parse an already-open icechunk Session directly.

        Bypasses the URL/Storage translation in ``__call__``. The session's
        snapshot is used as-is — the parser's ``branch``/``tag``/``snapshot_id``
        constructor args do not apply on this path.

        Parameters
        ----------
        session
            The open icechunk session to parse.
        registry
            ObjectStoreRegistry the resulting ManifestStore will use to read
            chunk data.
        native_chunks_prefix
            URL prefix to render icechunk's native (managed) chunk paths under.
            Native chunks become ``f"{native_chunks_prefix}/{chunk_id}"``.
            Required here — there's no URL for the parser to derive a default
            from. A single trailing slash is tolerated.
        """
        coro = _construct_manifest_group(
            store=session.store,
            group=self.group,
            native_chunks_prefix=native_chunks_prefix.rstrip("/"),
            skip_variables=self.skip_variables,
            batch_size=self.batch_size,
        )
        manifest_group = _run_async(coro)
        return ManifestStore(registry=registry, group=manifest_group)


async def _construct_manifest_group(
    store: "icechunk.IcechunkStore",
    *,
    native_chunks_prefix: str,
    group: str | None = None,
    skip_variables: Iterable[str] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ManifestGroup:
    """Build a ManifestGroup from an icechunk zarr group, recursing into subgroups."""
    zarr_group = await open_group_async(store=store, path=group, mode="r")
    skip = set() if skip_variables is None else set(skip_variables)

    array_keys = [key async for key in zarr_group.array_keys()]
    zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(k) for k in array_keys if k not in skip]
    )

    # zarr_group.getitem() returns AsyncArray | AsyncGroup; we filtered to
    # array_keys above, so every element is an AsyncArray in practice.
    manifest_arrays = await asyncio.gather(
        *[
            _construct_manifest_array(arr, store, native_chunks_prefix, batch_size)  # type: ignore[arg-type]
            for arr in zarr_arrays
        ]
    )
    arrays = {a.basename: ma for a, ma in zip(zarr_arrays, manifest_arrays)}

    # Subgroups, recursed depth-first so the full hierarchy is represented.
    # icechunk native-chunk ids are repo-global, so the same native_chunks_prefix
    # applies at every depth.
    group_keys = [key async for key in zarr_group.group_keys()]
    child_paths = [(key, key if not group else f"{group}/{key}") for key in group_keys]
    child_groups = await asyncio.gather(
        *[
            _construct_manifest_group(
                store=store,
                native_chunks_prefix=native_chunks_prefix,
                group=child_path,
                skip_variables=skip_variables,
                batch_size=batch_size,
            )
            for _, child_path in child_paths
        ]
    )
    groups = {name: mg for (name, _), mg in zip(child_paths, child_groups)}

    return ManifestGroup(
        arrays=arrays, groups=groups, attributes=dict(zarr_group.attrs)
    )


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
    # icechunk arrays use regular (uniform) chunk grids in practice; the union
    # member without chunk_shape (RectilinearChunkGridMetadata) doesn't apply.
    grid_shape = determine_chunk_grid_shape(
        metadata.shape,
        metadata.chunk_grid.chunk_shape,  # type: ignore[union-attr]
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

        flat_idx: np.ndarray
        if grid_shape:
            # np.ravel_multi_index's stubs pick the scalar overload here; the
            # array form is what we actually get from a (n_dims, n) input.
            flat_idx = np.asarray(np.ravel_multi_index(b_coords.T, grid_shape))
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
