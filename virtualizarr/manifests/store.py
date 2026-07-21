from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeAlias
from urllib.parse import urlparse

from obspec_utils.registry import ObjectStoreRegistry
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.common import BytesLike

from virtualizarr.manifests.array import ManifestArray
from virtualizarr.manifests.group import ManifestGroup
from virtualizarr.manifests.utils import parse_manifest_index

if TYPE_CHECKING:
    from obstore.store import (
        ObjectStore,
    )

    StoreDict: TypeAlias = dict[str, ObjectStore]

    import xarray as xr


__all__ = ["ManifestStore"]


@dataclass
class StoreRequest:
    """Dataclass for matching a key to the store instance"""

    store: ObjectStore
    """The ObjectStore instance to use for making the request."""
    key: str
    """The key within the store to request."""


@dataclass
class _ChunkRef:
    """A single chunk request resolved to an absolute byte range in a source file."""

    store: ObjectStore
    path: str
    start: int
    end: int


class _Member(NamedTuple):
    """One chunk request within a file group."""

    request_index: int
    """Position of this request in the original ``get_many`` ``requests``."""
    start: int
    """Start of the chunk's byte range, absolute within the source file."""
    end: int
    """End (exclusive) of the chunk's byte range, absolute within the source file."""


@dataclass
class _FileGroup:
    """A set of chunk requests that all read from the same source file."""

    store: ObjectStore
    path: str
    members: list[_Member]


def _coalesce_members(
    members: list[_Member], *, max_gap: int, max_bytes: int
) -> list[list[_Member]]:
    """Group members (all in the same file) into runs, each served by one read.

    Two members join the same run when the gap between them is at most
    ``max_gap`` *and* the resulting run span stays at most ``max_bytes``.

    The two knobs play distinct roles:

    - ``max_gap`` decides *whether* to bridge a gap between references, i.e. how
      much unwanted data coalescing is allowed to pull in. This is what governs
      strided access (a large gap bridges the stride of a 2D grid and over-reads;
      ``0`` merges only adjacent references and never over-reads).
    - ``max_bytes`` caps the *size* of any single read, so a long run of
      (near-)adjacent members is split into several bounded reads that the caller
      fetches concurrently. It bounds per-read memory and lets a large contiguous
      span be fetched over several parallel connections rather than one. It does
      *not* affect over-read.

    Runs are returned sorted by start offset and cover every member exactly
    once. A member larger than ``max_bytes`` still forms its own (single) run.
    """
    ordered = sorted(members, key=lambda member: member.start)
    runs: list[list[_Member]] = []
    run_start = run_end = 0
    for member in ordered:
        if (
            runs
            and member.start - run_end <= max_gap
            and max(run_end, member.end) - run_start <= max_bytes
        ):
            runs[-1].append(member)
            run_end = max(run_end, member.end)
        else:
            runs.append([member])
            run_start, run_end = member.start, member.end
    return runs


def get_store_prefix(url: str) -> str:
    """
    Get a logical prefix to use for a url in an ObjectStoreRegistry
    """
    scheme, netloc, *_ = urlparse(url)
    return "" if scheme in {"", "file"} else f"{scheme}://{netloc}"


def _get_deepest_group_or_array(
    node: ManifestGroup, key: str
) -> tuple[ManifestGroup | ManifestArray, str]:
    """
    Traverse the manifest hierarchy as deeply as possible following the given key path.

    Traversal stops when:
    - A key part doesn't match any array or group in the current node
    - A ManifestArray is reached (arrays cannot be traversed further)
    - All key parts have been successfully matched

    Args:
        node: The starting ManifestGroup to begin traversal from
        key: The key to use to traverse through groups and arrays

    Returns:
        A tuple containing:
        - The deepest node reached (ManifestGroup or ManifestArray)
        - String with remaining unmatched key portion
    """
    var, suffix = key.split("/", 1) if "/" in key else (key, "")
    if var in node.arrays:
        return node.arrays[var], suffix
    if var in node.groups:
        return _get_deepest_group_or_array(node.groups[var], suffix)
    # Can't traverse deeper - return last node and remainder
    return node, suffix or var


class ManifestStore(Store):
    """
    A read-only Zarr store that uses obstore to read data from inside arbitrary files on AWS, GCP, Azure, or a local filesystem.

    The requests from the Zarr API are redirected using the [ManifestGroup][virtualizarr.manifests.ManifestGroup] containing
    multiple [ManifestArray][virtualizarr.manifests.ManifestArray], allowing for virtually interfacing with underlying data in other file formats.

    Parameters
    ----------
    group
        Root group of the store.
        Contains group metadata, [ManifestArrays][virtualizarr.manifests.ManifestArray], and any subgroups.
    registry : ObjectStoreRegistry
        [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] that maps the URL scheme and netloc to  [ObjectStore][obstore.store.ObjectStore] instances,
        allowing ManifestStores to read from different ObjectStore instances.

    Warnings
    --------
    ManifestStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.
    """

    #  Modified from https://github.com/zarr-developers/zarr-python/pull/1661

    _group: ManifestGroup
    _registry: ObjectStoreRegistry

    def __eq__(self, value: object):
        NotImplementedError

    def __init__(
        self,
        group: ManifestGroup,
        *,
        registry: ObjectStoreRegistry | None = None,
        coalesce_max_gap_bytes: int = 0,
        coalesce_max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        """Instantiate a new ManifestStore.

        Parameters
        ----------
        group
            [ManifestGroup][virtualizarr.manifests.ManifestGroup] containing Group metadata and mapping variable names to ManifestArrays
        registry
            A registry mapping the URL scheme and netloc to  [ObjectStore][obstore.store.ObjectStore] instances,
            allowing [ManifestStores][virtualizarr.manifests.ManifestStore] to read from different  [ObjectStore][obstore.store.ObjectStore] instances.
        coalesce_max_gap_bytes
            When multiple chunks requested together (via ``get_many``) refer to the
            same source file, virtual references separated by at most this many bytes
            are coalesced into a single, larger read. Defaults to ``0``, which merges
            only references that are exactly adjacent in the file - a pure win, since
            no unwanted bytes are read. A larger value additionally bridges gaps up to
            that size, trading some wasted bytes for fewer requests; this can backfire
            for strided access patterns (e.g. a 2D spatial box, whose chunks are
            contiguous along one axis but far apart along another), where bridging the
            stride pulls in the intervening chunks. See
            [ManifestStore.get_many][virtualizarr.manifests.ManifestStore.get_many].
        coalesce_max_bytes
            Upper bound on the size of a single coalesced read. A run of adjacent (or,
            with a non-zero gap, near-adjacent) references longer than this is split
            into several reads that are fetched concurrently, bounding per-read memory
            and letting a large contiguous span be pulled over several parallel
            connections instead of one. This is the equivalent of icechunk's
            ``ideal_concurrent_request_size``, provided here because obstore does not
            split a single ranged read on its own. It does not affect over-read (that
            is ``coalesce_max_gap_bytes``), and is inert unless a query coalesces into a
            run larger than this. Defaults to 8 MiB.
        """

        if not isinstance(group, ManifestGroup):
            raise TypeError

        super().__init__(read_only=True)
        self._registry = ObjectStoreRegistry() if registry is None else registry
        self._group = group
        self._coalesce_max_gap_bytes = coalesce_max_gap_bytes
        self._coalesce_max_bytes = coalesce_max_bytes

    def __str__(self) -> str:
        return f"ManifestStore(group={self._group}, registry={self._registry})"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        node, suffix = _get_deepest_group_or_array(self._group, key)
        if suffix.endswith("zarr.json"):
            # Return metadata
            return node.metadata.to_buffer_dict(prototype=default_buffer_prototype())[
                "zarr.json"
            ]
        elif suffix.endswith((".zattrs", ".zgroup", ".zarray", ".zmetadata")):
            # Zarr-Python expects store classes to return None when metadata JSONs are not found.
            # Zarr-Python uses this behavior to distinguish between V2/V3 and consolidated/unconsolidated stores.
            # This upstream behavior will hopefully change in the future to be more Zarr-hierarchy aware, in
            # which case this may need refactoring.
            return None
        if isinstance(node, ManifestGroup):
            raise ValueError(
                "Key requested is a group but the key does not end in `zarr.json`"
            )
        manifest = node.manifest

        separator: Literal[".", "/"] = getattr(
            node.metadata.chunk_key_encoding, "separator", "."
        )
        chunk_indexes = parse_manifest_index(key, separator, expand_pattern=True)

        # Check for inlined (in-memory) chunks first
        if chunk_indexes in manifest._inlined:
            inlined_data = manifest._inlined[chunk_indexes]
            if byte_range is not None:
                inlined_byte_range = _transform_byte_range(
                    byte_range,
                    chunk_start=0,
                    chunk_end_exclusive=len(inlined_data),
                )
                inlined_data = inlined_data[
                    inlined_byte_range.start : inlined_byte_range.end
                ]
            return prototype.buffer.from_bytes(inlined_data)

        entry = manifest.get_entry(chunk_indexes)
        if entry is None:
            return None
        store, path_in_store = self._resolve_store_and_path(entry["path"])
        # Transform the input byte range to account for the chunk location in the file
        byte_range = _transform_byte_range(
            byte_range,
            chunk_start=entry["offset"],
            chunk_end_exclusive=entry["offset"] + entry["length"],
        )

        # Actually get the bytes
        bytes = await store.get_range_async(
            path_in_store,
            start=byte_range.start,
            end=byte_range.end,
        )
        return prototype.buffer.from_bytes(bytes)  # type: ignore[arg-type]

    def _resolve_store_and_path(self, path: str) -> tuple[ObjectStore, str]:
        """Resolve a manifest entry ``path`` to its ObjectStore and the path
        within that store (with any store prefix stripped)."""
        store, _ = self._registry.resolve(path)
        if not store:
            raise ValueError(
                f"Could not find a store to use for {path} in the store registry"
            )
        path_in_store = urlparse(path).path
        if hasattr(store, "prefix") and store.prefix:
            prefix = str(store.prefix).lstrip("/")
        elif hasattr(store, "url"):
            prefix = urlparse(store.url).path.lstrip("/")
        else:
            prefix = ""
        return store, path_in_store.lstrip("/").removeprefix(prefix).lstrip("/")

    def _resolve_chunk_ref(
        self, key: str, byte_range: ByteRequest | None
    ) -> _ChunkRef | None:
        """Resolve a chunk key to an absolute byte range within its source file.

        Returns ``None`` when the key is not a plain, manifest-backed chunk that
        can participate in coalescing - i.e. metadata documents, inlined chunks,
        missing chunks, or group keys. Those are handled individually by ``get``.
        """
        node, suffix = _get_deepest_group_or_array(self._group, key)
        if suffix.endswith(
            ("zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata")
        ) or isinstance(node, ManifestGroup):
            return None
        manifest = node.manifest
        separator: Literal[".", "/"] = getattr(
            node.metadata.chunk_key_encoding, "separator", "."
        )
        chunk_indexes = parse_manifest_index(key, separator, expand_pattern=True)
        if chunk_indexes in manifest._inlined:
            return None
        entry = manifest.get_entry(chunk_indexes)
        if entry is None:
            return None
        store, path_in_store = self._resolve_store_and_path(entry["path"])
        rng = _transform_byte_range(
            byte_range,
            chunk_start=entry["offset"],
            chunk_end_exclusive=entry["offset"] + entry["length"],
        )
        return _ChunkRef(store=store, path=path_in_store, start=rng.start, end=rng.end)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        # TODO: Implement using private functions from the upstream Zarr obstore integration
        raise NotImplementedError

    async def get_many(
        self,
        requests: Sequence[tuple[str, ByteRequest | None] | str],
        *,
        prototype: BufferPrototype,
    ) -> AsyncGenerator[Sequence[tuple[int, Buffer | None]], None]:
        """Retrieve many chunks at once, coalescing reads from the same file.

        Overrides [Store.get_many][zarr.abc.store.Store.get_many]. Requested
        chunks are resolved through the manifests to ``(source file, byte
        range)`` and grouped by source file. Within each file, references closer
        together than ``coalesce_max_gap_bytes`` are coalesced into runs; each run
        is then split into reads of at most ``coalesce_max_bytes`` and served by a
        ranged read that is sliced back into per-chunk buffers. The gap controls
        how much unwanted data coalescing may pull in - ``0`` (the default) merges
        only adjacent references and never over-reads, which matters for strided
        access such as a 2D spatial box - while the size cap only bounds how large
        any single read may get (splitting big contiguous runs into parallel
        reads); it does not affect over-read.
        This is the same technique object_store and async-tiff use to read many
        tiles efficiently, applied here to virtual chunk references.

        Keys that are not plain manifest-backed chunks (metadata documents,
        inlined chunks, or missing chunks) are served individually via ``get``.
        Results are yielded as ``(request_index, Buffer | None)`` batches in
        completion order, per the ``Store.get_many`` contract.
        """
        # Local import so importing this module doesn't require the zarr config.
        from zarr.core.config import config

        # Partition requests into coalescable chunk reads (grouped by source
        # file) and everything else (served one-by-one via ``get``).
        groups: dict[tuple[int, str], _FileGroup] = {}
        singletons: list[tuple[int, str, ByteRequest | None]] = []
        for index, request in enumerate(requests):
            key, byte_range = (request, None) if isinstance(request, str) else request
            ref = self._resolve_chunk_ref(key, byte_range)
            if ref is None:
                singletons.append((index, key, byte_range))
                continue
            group = groups.setdefault(
                (id(ref.store), ref.path), _FileGroup(ref.store, ref.path, [])
            )
            group.members.append(_Member(index, ref.start, ref.end))

        semaphore = asyncio.Semaphore(config.get("async.concurrency"))

        async def fetch_run(
            store: ObjectStore, path: str, run: list[_Member]
        ) -> Sequence[tuple[int, Buffer | None]]:
            # One ranged read spanning the whole run, sliced back per chunk.
            run_start = run[0].start
            run_end = max(member.end for member in run)
            async with semaphore:
                data = await store.get_range_async(path, start=run_start, end=run_end)
            view = memoryview(data)  # type: ignore[arg-type]
            return [
                (
                    member.request_index,
                    prototype.buffer.from_bytes(
                        view[member.start - run_start : member.end - run_start]
                    ),
                )
                for member in run
            ]

        async def fetch_single(
            index: int, key: str, byte_range: ByteRequest | None
        ) -> Sequence[tuple[int, Buffer | None]]:
            async with semaphore:
                buffer = await self.get(key, prototype, byte_range)
            return ((index, buffer),)

        tasks = [
            asyncio.ensure_future(fetch_run(group.store, group.path, run))
            for group in groups.values()
            for run in _coalesce_members(
                group.members,
                max_gap=self._coalesce_max_gap_bytes,
                max_bytes=self._coalesce_max_bytes,
            )
        ]
        tasks += [
            asyncio.ensure_future(fetch_single(index, key, byte_range))
            for index, key, byte_range in singletons
        ]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    async def exists(self, key: str) -> bool:
        # docstring inherited
        raise NotImplementedError

    @property
    def supports_writes(self) -> bool:
        # docstring inherited
        return False

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        raise NotImplementedError

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        raise NotImplementedError

    @property
    def supports_deletes(self) -> bool:
        # docstring inherited
        return False

    @property
    def supports_partial_writes(self) -> Literal[False]:
        # docstring inherited
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        # docstring inherited
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        # docstring inherited
        return True

    def list(self) -> AsyncGenerator[str, None]:
        # docstring inherited
        raise NotImplementedError

    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        raise NotImplementedError

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        # Navigate to the target node
        node, suffix = _get_deepest_group_or_array(self._group, prefix)
        # Zarr-Python lists using a per-path basis, so we don't have anything to list
        # as long as there is a suffix remaining and we require a '.' chunk separator in the ManifestArrays
        if suffix:
            return
        # List contents based on node type
        if isinstance(node, ManifestGroup):
            # Groups contain a metadata document and the name of sub-groups/arrays
            yield "zarr.json"
            for member_name in node._members.keys():
                yield member_name
        # TODO: Support listing when using other chunk_key_encodings
        elif (
            separator := getattr(node.metadata.chunk_key_encoding, "separator", None)
            != "."
        ):
            raise NotImplementedError(
                f"Array listing only supports '.' as chunk key separator, "
                f"got {separator!r}"
            )
        else:
            # Arrays contain a metadata document and chunks
            yield "zarr.json"
            if node.shape == ():
                # Scalar arrays have a single chunk named 'c'
                yield "c"
            else:
                # Multi-dimensional arrays have chunks named 'c.{key}'
                for chunk_key in node.manifest.keys():
                    yield f"c.{chunk_key}"

    @property
    def supports_consolidated_metadata(self) -> bool:
        # docstring inherited
        return False

    def to_virtual_dataset(
        self,
        group="",
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
    ) -> "xr.Dataset":
        """
        Create a "virtual" [xarray.Dataset][] containing the contents of one zarr group.

        Will ignore the contents of any other groups in the store.

        Requires xarray.

        Parameters
        ----------
        group : str
        loadable_variables : Iterable[str], optional

        Returns
        -------
        vds : xarray.Dataset
        """

        from virtualizarr.xarray import construct_virtual_dataset

        if loadable_variables and self._registry.map is None:
            raise ValueError(
                f"ManifestStore contains an empty store registry, but {loadable_variables} were provided as loadable variables. Must provide an ObjectStore instance in order to load variables."
            )

        vds = construct_virtual_dataset(
            manifest_store=self,
            group=group,
            loadable_variables=loadable_variables,
            decode_times=decode_times,
        )
        _warn_about_oversized_virtual_chunks(vds)
        return vds

    def to_virtual_datatree(
        self,
        group="",
        *,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
    ) -> "xr.DataTree":
        """
        Create a "virtual" [xarray.DataTree][] containing the contents of a zarr group. Default is the root group and all sub-groups.

        Will ignore the contents of any other groups in the store.

        Requires xarray.

        Parameters
        ----------
        group : Group to convert to a virtual DataTree
        loadable_variables
            Variables in the data source to load as Dask/NumPy arrays instead of as virtual arrays.
        decode_times
            Bool that is passed into [xarray.open_dataset][]. Allows time to be decoded into a datetime object.

        Returns
        -------
        vdt : xarray.DataTree
        """

        from virtualizarr.xarray import construct_virtual_datatree

        return construct_virtual_datatree(
            manifest_store=self,
            group=group,
            loadable_variables=loadable_variables,
            decode_times=decode_times,
        )


def _warn_about_oversized_virtual_chunks(vds: "xr.Dataset") -> None:
    """
    Warn if any still-virtual variable has a chunk larger than its array shape.

    Such chunks arise for variables along an unlimited dimension whose oversized
    chunk could not be trimmed (e.g. because it is compressed). They read and
    write as virtual references fine, but cannot be concatenated with other
    virtual datasets - the oversized final chunk prevents forming a regular
    chunk grid - so point the user at loading them instead. Variables that were
    loaded (no longer backed by a ManifestArray) are unaffected and not warned
    about.
    """
    oversized = [
        name
        for name, var in vds.variables.items()
        if isinstance(var.data, ManifestArray)
        and any(c > s for c, s in zip(var.data.metadata.chunks, var.data.shape))
    ]
    if oversized:
        warnings.warn(
            f"Variable(s) {oversized} have a chunk shape larger than their array "
            "shape, which typically happens for variables along an unlimited "
            "dimension. They read and write as virtual references correctly, but "
            "cannot be concatenated with other virtual datasets because the "
            "oversized chunk prevents forming a regular chunk grid. Pass them to "
            "loadable_variables to load them as in-memory arrays if you need to "
            "concatenate them.",
            UserWarning,
            stacklevel=3,
        )


def _transform_byte_range(
    byte_range: ByteRequest | None, *, chunk_start: int, chunk_end_exclusive: int
) -> RangeByteRequest:
    """
    Convert an incoming byte_range which assumes one chunk per file to a
    virtual byte range that accounts for the location of a chunk within a file.
    """
    if byte_range is None:
        byte_range = RangeByteRequest(chunk_start, chunk_end_exclusive)
    elif isinstance(byte_range, RangeByteRequest):
        if byte_range.end > chunk_end_exclusive:
            raise ValueError(
                f"Chunk ends before byte {chunk_end_exclusive} but request end was {byte_range.end}"
            )
        byte_range = RangeByteRequest(
            chunk_start + byte_range.start, chunk_start + byte_range.end
        )
    elif isinstance(byte_range, OffsetByteRequest):
        byte_range = RangeByteRequest(
            chunk_start + byte_range.offset, chunk_end_exclusive
        )  # type: ignore[arg-type]
    elif isinstance(byte_range, SuffixByteRequest):
        byte_range = RangeByteRequest(
            chunk_end_exclusive - byte_range.suffix, chunk_end_exclusive
        )  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unexpected byte_range, got {byte_range}")
    return byte_range
