from __future__ import annotations

import asyncio
import math
import struct
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from obspec_utils.registry import ObjectStoreRegistry
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.metadata import ArrayV2Metadata

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestStore,
)
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.manifests.utils import ChunkKeySeparator
from virtualizarr.parsers.utils import construct_manifest_group_tree
from virtualizarr.parsers.zarr import (
    ObstoreStore,
    RegularChunkGridMetadata,
    ZarrFormat,
    _run_async,
    join_url,
    metadata_as_v3,
)
from virtualizarr.utils import determine_chunk_grid_shape

ZIP_METHOD_STORED = 0

_ZIP_METHOD_NAMES = {
    0: "STORED",
    8: "DEFLATE",
    9: "DEFLATE64",
    12: "BZIP2",
    14: "LZMA",
    93: "ZSTD",
}

_EOCD_SIG = b"PK\x05\x06"
_EOCD_LEN = 22
_ZIP64_EOCD_SIG = b"PK\x06\x06"
_ZIP64_EOCD_LEN = 56
_ZIP64_LOCATOR_SIG = b"PK\x06\x07"
_ZIP64_LOCATOR_LEN = 20
_CEN_SIG = b"PK\x01\x02"
_CEN_LEN = 46
_LOC_SIG = b"PK\x03\x04"
_LOC_LEN = 30
_MAX_COMMENT_LEN = 0xFFFF


@dataclass(frozen=True)
class ZipEntry:
    """Location and compression info for one member of a zip archive."""

    data_offset: int
    compressed_length: int
    uncompressed_length: int
    method: int

    @property
    def method_name(self) -> str:
        return _ZIP_METHOD_NAMES.get(self.method, f"method {self.method}")


def _parse_zip64_extra(
    extra: bytes, uncompressed: int, compressed: int, header_offset: int
) -> tuple[int, int, int]:
    """Resolve 0xFFFFFFFF sentinel fields from the ZIP64 extended information extra field."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        if header_id == 0x0001:
            # ZIP64 field contains 8-byte values only for the fields that overflowed,
            # in the fixed order: uncompressed size, compressed size, header offset
            values = extra[pos + 4 : pos + 4 + data_size]
            vpos = 0
            if uncompressed == 0xFFFFFFFF:
                uncompressed = struct.unpack_from("<Q", values, vpos)[0]
                vpos += 8
            if compressed == 0xFFFFFFFF:
                compressed = struct.unpack_from("<Q", values, vpos)[0]
                vpos += 8
            if header_offset == 0xFFFFFFFF:
                header_offset = struct.unpack_from("<Q", values, vpos)[0]
            break
        pos += 4 + data_size
    return uncompressed, compressed, header_offset


async def parse_zip_index(store: ObstoreStore, path: str) -> dict[str, ZipEntry]:
    """Read a zip archive's central directory and local headers to build a member index.

    Costs three or four range reads regardless of archive size: the tail (which usually
    contains the whole central directory), possibly the central directory itself, and
    one batched read of the members' local headers (whose extra-field lengths may
    legally differ from the central directory's, shifting the data start).

    Parameters
    ----------
    store
        An object store implementing the obspec ``HeadAsync``, ``GetRangeAsync`` and
        ``GetRangesAsync`` protocols.
    path
        The path of the zip archive within ``store``.

    Returns
    -------
    Mapping of member name to [ZipEntry][virtualizarr.parsers.zip.ZipEntry], in
    central directory order. Directory placeholder members are omitted.
    """
    file_size = (await store.head_async(path))["size"]

    tail_len = min(
        file_size,
        _EOCD_LEN + _MAX_COMMENT_LEN + _ZIP64_LOCATOR_LEN + _ZIP64_EOCD_LEN,
    )
    tail_start = file_size - tail_len
    tail = bytes(await store.get_range_async(path, start=tail_start, end=file_size))

    eocd_pos = tail.rfind(_EOCD_SIG)
    if eocd_pos == -1:
        raise ValueError(
            f"{path} is not a zip archive: no end-of-central-directory record found"
        )
    n_entries, cd_size, cd_offset = struct.unpack_from("<HII", tail, eocd_pos + 10)

    if 0xFFFF in (n_entries,) or 0xFFFFFFFF in (cd_size, cd_offset):
        locator_pos = tail.rfind(_ZIP64_LOCATOR_SIG, 0, eocd_pos)
        if locator_pos == -1:
            raise ValueError(f"{path}: ZIP64 sentinel values but no ZIP64 locator")
        zip64_eocd_offset = struct.unpack_from("<Q", tail, locator_pos + 8)[0]
        if zip64_eocd_offset >= tail_start:
            zip64_eocd = tail[zip64_eocd_offset - tail_start :]
        else:
            zip64_eocd = bytes(
                await store.get_range_async(
                    path,
                    start=zip64_eocd_offset,
                    end=zip64_eocd_offset + _ZIP64_EOCD_LEN,
                )
            )
        if not zip64_eocd.startswith(_ZIP64_EOCD_SIG):
            raise ValueError(f"{path}: invalid ZIP64 end-of-central-directory record")
        n_entries, cd_size, cd_offset = struct.unpack_from("<QQQ", zip64_eocd, 32)

    if cd_offset >= tail_start:
        cd = tail[cd_offset - tail_start : cd_offset - tail_start + cd_size]
    else:
        cd = bytes(
            await store.get_range_async(path, start=cd_offset, end=cd_offset + cd_size)
        )

    # Walk the central directory records
    names: list[str] = []
    methods: list[int] = []
    compressed_lengths: list[int] = []
    uncompressed_lengths: list[int] = []
    header_offsets: list[int] = []
    pos = 0
    while pos + _CEN_LEN <= len(cd) and cd[pos : pos + 4] == _CEN_SIG:
        (
            _sig,
            _version_made_by,
            _version_needed,
            _flags,
            method,
            _mtime,
            _mdate,
            _crc,
            compressed,
            uncompressed,
            fn_len,
            extra_len,
            comment_len,
            _disk_number,
            _internal_attrs,
            _external_attrs,
            header_offset,
        ) = struct.unpack_from("<IHHHHHHIIIHHHHHII", cd, pos)
        name = cd[pos + _CEN_LEN : pos + _CEN_LEN + fn_len].decode("utf-8")
        if 0xFFFFFFFF in (uncompressed, compressed, header_offset):
            extra = cd[pos + _CEN_LEN + fn_len : pos + _CEN_LEN + fn_len + extra_len]
            uncompressed, compressed, header_offset = _parse_zip64_extra(
                extra, uncompressed, compressed, header_offset
            )
        pos += _CEN_LEN + fn_len + extra_len + comment_len

        if name.endswith("/") and uncompressed == 0:
            # directory placeholder entry, not a member
            continue
        names.append(name)
        methods.append(method)
        compressed_lengths.append(compressed)
        uncompressed_lengths.append(uncompressed)
        header_offsets.append(header_offset)

    # directory placeholder members legitimately reduce the count below n_entries
    if len(names) > n_entries:
        raise ValueError(
            f"{path}: central directory holds {len(names)} entries, expected {n_entries}"
        )

    if not names:
        return {}

    # The local header's filename/extra-field lengths determine where member data
    # actually starts, and the extra field may differ in length from the central
    # directory's copy — so read each member's 30-byte local header, in one batched request.
    local_headers = await store.get_ranges_async(
        path,
        starts=header_offsets,
        ends=[offset + _LOC_LEN for offset in header_offsets],
    )

    index: dict[str, ZipEntry] = {}
    for name, method, compressed, uncompressed, header_offset, local_header in zip(
        names,
        methods,
        compressed_lengths,
        uncompressed_lengths,
        header_offsets,
        local_headers,
    ):
        local = bytes(local_header)
        if not local.startswith(_LOC_SIG):
            raise ValueError(f"{path}: invalid local file header for member {name!r}")
        local_fn_len, local_extra_len = struct.unpack_from("<HH", local, 26)
        index[name] = ZipEntry(
            data_offset=header_offset + _LOC_LEN + local_fn_len + local_extra_len,
            compressed_length=compressed,
            uncompressed_length=uncompressed,
            method=method,
        )
    return index


class _ZipMetadataStore(Store):
    """Read-only zarr store presenting the members of a zip archive.

    Used only to read zarr metadata documents while parsing: chunk members never pass
    through this store, they become manifest entries pointing directly at byte ranges
    within the archive.
    """

    def __init__(
        self, store: ObstoreStore, path: str, index: Mapping[str, ZipEntry]
    ) -> None:
        super().__init__(read_only=True)
        self._store = store
        self._path = path
        self._index = index

    def __eq__(self, other: object) -> bool:
        return self is other

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        entry = self._index.get(key)
        if entry is None:
            return None
        start = entry.data_offset
        end = entry.data_offset + entry.compressed_length
        if isinstance(byte_range, RangeByteRequest):
            end = min(entry.data_offset + byte_range.end, end)
            start = entry.data_offset + byte_range.start
        elif isinstance(byte_range, OffsetByteRequest):
            start = entry.data_offset + byte_range.offset
        elif isinstance(byte_range, SuffixByteRequest):
            start = end - byte_range.suffix
        data = await self._store.get_range_async(self._path, start=start, end=end)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return await asyncio.gather(
            *[self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
        )

    async def exists(self, key: str) -> bool:
        return key in self._index

    @property
    def supports_writes(self) -> bool:
        return False

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError

    @property
    def supports_deletes(self) -> bool:
        return False

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        for key in self._index:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        for key in self._index:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        prefix = prefix.rstrip("/")
        seen: set[str] = set()
        for key in self._index:
            if prefix and not key.startswith(prefix + "/"):
                continue
            remainder = key[len(prefix) + 1 :] if prefix else key
            child = remainder.split("/")[0]
            if child and child not in seen:
                seen.add(child)
                yield child


async def _construct_manifest_array(
    zarr_array: Any,
    *,
    zip_uri: str,
    index: Mapping[str, ZipEntry],
) -> ManifestArray:
    """Build a ManifestArray for one array, sourcing chunk locations from the zip index."""
    metadata = metadata_as_v3(zarr_array.metadata)

    if not isinstance(metadata.chunk_grid, RegularChunkGridMetadata):
        raise NotImplementedError(
            f"Only RegularChunkGrid is supported, but array {zarr_array.path} "
            f"uses {type(metadata.chunk_grid).__name__}."
        )

    on_disk_format = ZarrFormat(zarr_array.metadata.zarr_format)
    separator: ChunkKeySeparator = (
        zarr_array.metadata.chunk_key_encoding.separator
        if on_disk_format == ZarrFormat.V3
        else cast(ArrayV2Metadata, zarr_array.metadata).dimension_separator
    )

    chunk_grid_shape = determine_chunk_grid_shape(
        metadata.shape,
        cast(RegularChunkGridMetadata, metadata.chunk_grid).chunk_shape,
    )

    array_path = zarr_array.path

    if metadata.shape == ():
        scalar_key = join_url(array_path, on_disk_format.scalar_chunk_key_name)
        entry = index.get(scalar_key)
        if entry is None:
            # The zarr spec allows the scalar chunk to be uninitialized
            chunk_manifest = ChunkManifest({}, shape=chunk_grid_shape)
        else:
            chunk_manifest = ChunkManifest(
                {
                    "c": {
                        "path": zip_uri,
                        "offset": entry.data_offset,
                        "length": entry.compressed_length,
                    }
                }
            )
        return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)

    chunks_prefix = join_url(array_path, on_disk_format.chunks_dir_prefix)
    if not chunks_prefix.endswith("/"):
        chunks_prefix += "/"

    chunk_keys: list[str] = []
    offsets: list[int] = []
    lengths: list[int] = []
    for member, entry in index.items():
        if not member.startswith(chunks_prefix):
            continue
        if member.endswith(on_disk_format.metadata_key_names):
            continue
        chunk_keys.append(member[len(chunks_prefix) :])
        offsets.append(entry.data_offset)
        lengths.append(entry.compressed_length)

    if not chunk_keys:
        return ManifestArray(
            metadata=metadata, chunkmanifest=ChunkManifest({}, shape=chunk_grid_shape)
        )

    coords = np.array(
        [[int(c) for c in key.split(separator)] for key in chunk_keys],
        dtype=np.int64,
    ).T  # shape: (ndim, nchunks)
    flat_positions = np.ravel_multi_index(coords, chunk_grid_shape)

    total_size = math.prod(chunk_grid_shape)
    dense_paths = np.full(total_size, "", dtype=np.dtypes.StringDType())
    dense_offsets = np.zeros(total_size, dtype=np.uint64)
    dense_lengths = np.zeros(total_size, dtype=np.uint64)
    dense_paths[flat_positions] = zip_uri
    dense_offsets[flat_positions] = offsets
    dense_lengths[flat_positions] = lengths

    chunk_manifest = ChunkManifest.from_arrays(
        paths=dense_paths.reshape(chunk_grid_shape),
        offsets=dense_offsets.reshape(chunk_grid_shape),
        lengths=dense_lengths.reshape(chunk_grid_shape),
    )
    return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)


class ZippedZarrParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from a zipped Zarr store (e.g. a ``.zarr.zip`` file).

    Creates lightweight virtual references to chunks inside the zip archive without
    extracting or copying data: the archive's central directory locates every member,
    and each chunk becomes a byte range within the archive itself. Supports both Zarr
    V2 and V3 formats, automatically converting V2 metadata to V3.

    Only archives whose members are stored without compression (the zip ``STORED``
    method — ``zip -0``, and the default for ``zarr.storage.ZipStore``) are currently
    supported: compressed members do not preserve chunk byte ranges. Support for
    ``DEFLATE``-compressed zipped Zarr, and for wrapping other parsers to read
    non-Zarr formats inside zip archives, may be added in the future.

    Parameters
    ----------
    group
        Path to a specific group within the zipped Zarr store to be used as the Zarr
        root group for the ManifestStore. Uses forward slashes for nested groups
        (e.g., "model/output"). Default is None, which uses the store's root group.
    skip_variables
        Variables in the zipped Zarr store that will be ignored when creating the
        ManifestStore. Default is None, which includes all variables.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets of a zipped Zarr store to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the zip archive (e.g. ``"s3://bucket/store.zarr.zip"``).
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for
            resolving urls and reading data.

        Returns
        -------
        [ManifestStore][virtualizarr.manifests.ManifestStore]
            A virtual representation of the zipped Zarr store with references to
            byte ranges within the archive.

        Raises
        ------
        ValueError
            If the URL cannot be resolved, or the file is not a valid zip archive.
        NotImplementedError
            If any archive member is compressed (only ``STORED`` members are supported).
        """
        uri = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())

        object_store, store_relative_path = registry.resolve(uri)
        archive_path = str(store_relative_path)

        index = _run_async(parse_zip_index(object_store, archive_path))

        compressed = {
            name: entry
            for name, entry in index.items()
            if entry.method != ZIP_METHOD_STORED
        }
        if compressed:
            example_name, example_entry = next(iter(compressed.items()))
            raise NotImplementedError(
                f"{len(compressed)} of {len(index)} members of {url} are compressed "
                f"(e.g. {example_name!r} uses {example_entry.method_name}), so their chunks are not "
                f"contiguous byte ranges within the archive. Only archives with "
                f"uncompressed (STORED) members can currently be virtualized: recreate "
                f"the archive without compression (e.g. `zip -0`)."
            )

        metadata_store = _ZipMetadataStore(object_store, archive_path, index)

        manifest_group = _run_async(
            construct_manifest_group_tree(
                metadata_store,
                build_manifest_array=lambda zarr_array: _construct_manifest_array(
                    zarr_array, zip_uri=uri, index=index
                ),
                group=self.group,
                skip_variables=self.skip_variables,
            )
        )

        return ManifestStore(registry=registry, group=manifest_group)
