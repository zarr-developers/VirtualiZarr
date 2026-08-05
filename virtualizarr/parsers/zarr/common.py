"""Helpers shared by the parsers for Zarr-format stores (ZarrParser and ZippedZarrParser)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
from collections.abc import Coroutine, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import zarr
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from virtualizarr.manifests import ChunkManifest
from virtualizarr.manifests.utils import ChunkKeySeparator
from virtualizarr.utils import determine_chunk_grid_shape

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import RegularChunkGridMetadata
else:
    try:
        from zarr.core.metadata.v3 import RegularChunkGridMetadata  # zarr-python>3.1.6
    except ImportError:
        from zarr.core.metadata.v3 import (
            RegularChunkGrid as RegularChunkGridMetadata,  # zarr-python<=3.1.6
        )

# obstore doesn't export a public base type for stores, so we use Any for now.
ObstoreStore = Any

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine, handling the case where an event loop is already running.

    In environments like Jupyter notebooks, an event loop is already running,
    so ``asyncio.run()`` raises ``RuntimeError``. In that case we run the
    coroutine in a separate thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop – the simple path.
        return asyncio.run(coro)

    # A loop is already running (e.g. Jupyter).  Execute in a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class ZarrFormat(Enum):
    """
    Encode all differences between on-disk Zarr formats here.

    Note that we still only need to support the zarr-python v3 API,
    so this enum is only concerned with differences in the native format spec between versions.
    """

    V2 = 2
    V3 = 3

    @property
    def metadata_key_names(self) -> tuple[str, ...]:
        match self:
            case ZarrFormat.V2:
                return (".zarray", ".zattrs", ".zgroup", ".zmetadata")
            case ZarrFormat.V3:
                return ("zarr.json",)

    @property
    def scalar_chunk_key_name(self) -> str:
        match self:
            case ZarrFormat.V2:
                return "0"
            case ZarrFormat.V3:
                return "c"

    def chunks_prefix(self, separator: ChunkKeySeparator) -> str:
        """
        Prefix that every non-scalar chunk key of an array shares, relative to the
        array's own path.

        V3 keys are prefixed with ``"c"`` joined to the coordinates by the array's chunk
        key separator, so this is a directory only when that separator is ``"/"``
        (e.g. ``"c/"`` for ``air/c/0/0``, but ``"c."`` for ``air/c.0.0``). V2 keys carry
        no prefix at all.
        """
        match self:
            case ZarrFormat.V2:
                return ""
            case ZarrFormat.V3:
                return f"c{separator}"


def join_url(base: str, key: str) -> str:
    """Join a base URL (like s3://bucket/store.zarr) with an object key.

    Ensures we don't accidentally produce double slashes (after the scheme)
    and that the returned string is scheme-friendly.
    """
    if not base:
        return key
    # strip trailing slash from base and leading slash from key to avoid '//' in middle
    return base.rstrip("/") + "/" + key.lstrip("/")


def metadata_as_v3(metadata: ArrayV3Metadata | ArrayV2Metadata) -> ArrayV3Metadata:
    """Convert metadata to V3 format with normalized chunk_key_encoding."""

    if isinstance(metadata, ArrayV2Metadata):
        v3_dict = _convert_v2_to_v3_dict(metadata)
    else:
        v3_dict = metadata.to_dict()

    # Normalize chunk_key_encoding to DefaultChunkKeyEncoding with "." separator.
    # The ManifestStore expects dot-separated keys (e.g. "0.0.0"), so we enforce
    # this regardless of what the on-disk store uses.
    v3_dict["chunk_key_encoding"] = {"name": "default", "separator": "."}
    return ArrayV3Metadata.from_dict(v3_dict)


def _convert_v2_to_v3_dict(metadata: ArrayV2Metadata) -> dict:
    """Convert V2 metadata to a V3 dict, handling fill_value, dimensions, and attributes."""

    try:
        from zarr.core.dtype import parse_dtype
        from zarr.metadata.migrate_v3 import _convert_array_metadata
    except (ImportError, AttributeError):
        raise ImportError(
            f"Zarr-Python>=3.1.3 is required for parsing Zarr V2 into Zarr V3. "
            f"Found Zarr version '{zarr.__version__}'"
        )

    # V3 requires a non-None fill_value, but V2 allows it. If missing, set to the
    # dtype's default (e.g. 0 for int) before converting. We roundtrip through a dict
    # because ArrayV2Metadata is immutable.
    if metadata.fill_value is None:
        v2_dict = metadata.to_dict()
        v2_dtype = parse_dtype(cast(Any, v2_dict["dtype"]), zarr_format=2)
        fill_value = v2_dtype.default_scalar()
        v2_dict["fill_value"] = v2_dtype.to_json_scalar(fill_value, zarr_format=2)
        metadata = ArrayV2Metadata.from_dict(v2_dict)

    v3_dict = _convert_array_metadata(metadata).to_dict()

    # _convert_array_metadata doesn't promote V2's _ARRAY_DIMENSIONS attribute
    # to V3's dimension_names, so we do it manually.
    attrs = cast(dict, v3_dict.get("attributes", {}))
    dim_names = attrs.get("_ARRAY_DIMENSIONS")
    if v3_dict.get("dimension_names") is None and dim_names is not None:
        v3_dict["dimension_names"] = dim_names

    # _ARRAY_DIMENSIONS is a V2 convention that gets promoted to dimension_names in V3,
    # so remove it from attributes to avoid duplication.
    if "_ARRAY_DIMENSIONS" in attrs:
        del attrs["_ARRAY_DIMENSIONS"]

    return v3_dict


def parse_array_layout(
    zarr_array: zarr.AsyncArray[Any],
) -> tuple[ArrayV3Metadata, ZarrFormat, ChunkKeySeparator, tuple[int, ...]]:
    """Extract what any parser needs to build a ManifestArray from an opened zarr array.

    Returns
    -------
    Tuple of (normalized V3 metadata, on-disk zarr format, on-disk chunk key separator,
    chunk grid shape).
    """
    metadata = metadata_as_v3(zarr_array.metadata)

    if not isinstance(metadata.chunk_grid, RegularChunkGridMetadata):
        raise NotImplementedError(
            f"Only RegularChunkGrid is supported, but array {zarr_array.path} "
            f"uses {type(metadata.chunk_grid).__name__}."
        )

    # The on-disk format determines how chunks are stored (e.g. V2 has no c/ prefix),
    # which differs from the always-V3 metadata we use internally.
    on_disk_zarr_format = ZarrFormat(zarr_array.metadata.zarr_format)
    on_disk_separator: ChunkKeySeparator = (
        zarr_array.metadata.chunk_key_encoding.separator
        if on_disk_zarr_format == ZarrFormat.V3
        else cast(ArrayV2Metadata, zarr_array.metadata).dimension_separator
    )

    # For sharded arrays, chunk_grid.chunk_shape is the shard shape (not the inner
    # chunk shape, which lives inside the ShardingCodec config). So this grid describes
    # the number of shard files on disk, which is exactly what we want for the manifest.
    chunk_grid_shape = determine_chunk_grid_shape(
        metadata.shape, cast(RegularChunkGridMetadata, metadata.chunk_grid).chunk_shape
    )

    return metadata, on_disk_zarr_format, on_disk_separator, chunk_grid_shape


def chunk_entries_to_manifest(
    chunk_keys: np.ndarray | Sequence[str],
    paths: np.ndarray | str,
    offsets: np.ndarray | Sequence[int],
    lengths: np.ndarray | Sequence[int],
    *,
    separator: ChunkKeySeparator,
    chunk_grid_shape: tuple[int, ...],
) -> ChunkManifest:
    """Scatter sparse chunk entries into a dense ChunkManifest.

    Chunks absent from ``chunk_keys`` are left uninitialized in the manifest, preserving
    sparsity (zarr returns the fill_value for those regions when the array is read).

    Parameters
    ----------
    chunk_keys
        Separator-delimited grid coordinates, one per initialized chunk (e.g. "0.0.0").
    paths
        Full URI per chunk, or a single URI shared by every chunk.
    offsets
        Byte offset within the file per chunk.
    lengths
        Byte length per chunk.
    separator
        The chunk key separator used in ``chunk_keys`` (e.g. ``"."`` or ``"/"``).
    chunk_grid_shape
        Shape of the array's chunk grid.
    """
    if len(chunk_keys) == 0:
        return ChunkManifest({}, shape=chunk_grid_shape)

    # split "0.0.0" style keys into per-dimension integer coords
    # TODO replace np.char.split with np.strings.split once it exists
    split_keys = np.char.split(np.asarray(chunk_keys), sep=separator)
    coords = np.array(
        [[int(c) for c in key] for key in split_keys], dtype=np.int64
    ).T  # shape: (ndim, nchunks)
    flat_positions = np.ravel_multi_index(coords, chunk_grid_shape)

    # scatter listed chunks into dense flat arrays (empty string / 0 = missing)
    total_size = math.prod(chunk_grid_shape)
    dense_paths = np.full(total_size, "", dtype=np.dtypes.StringDType())
    dense_offsets = np.zeros(total_size, dtype=np.uint64)
    dense_lengths = np.zeros(total_size, dtype=np.uint64)
    dense_paths[flat_positions] = paths
    dense_offsets[flat_positions] = offsets
    dense_lengths[flat_positions] = lengths

    return ChunkManifest.from_arrays(
        paths=dense_paths.reshape(chunk_grid_shape),
        offsets=dense_offsets.reshape(chunk_grid_shape),
        lengths=dense_lengths.reshape(chunk_grid_shape),
    )
