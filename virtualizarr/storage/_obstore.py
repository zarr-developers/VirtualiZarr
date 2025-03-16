from __future__ import annotations

import asyncio
import pickle
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

from xarray import DataArray, Dataset
from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

from virtualizarr.storage.common import (
    find_matching_store,
    get_zarr_metadata,
    list_dir_from_xr_obj,
    parse_manifest_index,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable
    from typing import Any

    from obstore import OffsetRange, SuffixRange
    from obstore.store import ObjectStore as _UpstreamObjectStore
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import BytesLike

    from virtualizarr.types.general import T_Xarray


__all__ = ["VirtualObjectStore"]

_ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)


class VirtualObjectStore(Store):
    """A Zarr store that uses obstore for fast read/write from AWS, GCP, Azure.

    Parameters
    ----------
    stores : dict[prefix, obstore.store.ObjectStore]
        A mapping of url prefixes to obstore store instance set up with the proper credentials.

    Warnings
    --------
    ObjectStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.

    Notes
    -----
    Modified from https://github.com/zarr-developers/zarr-python/pull/1661
    """

    def __eq__(self, value: object) -> bool:
        NotImplementedError

    def __init__(
        self,
        xr_obj: T_Xarray,
        stores: dict[str:_UpstreamObjectStore],
    ) -> None:
        import obstore as obs

        # TODO: Support DataArray, Dataset, or DataTree across all methods
        for store in stores.values():
            if not isinstance(
                store,
                (
                    obs.store.AzureStore,
                    obs.store.GCSStore,
                    obs.store.HTTPStore,
                    obs.store.S3Store,
                    obs.store.LocalStore,
                    obs.store.MemoryStore,
                ),
            ):
                raise TypeError(f"expected ObjectStore class, got {store!r}")
        super().__init__(read_only=True)
        self.stores = stores
        self.xr_obj = xr_obj

    def __str__(self) -> str:
        return f"ManifesStore({self.xr_obj})"

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        state["store"] = pickle.dumps(self.store)
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        state["store"] = pickle.loads(state["store"])
        self.__dict__.update(state)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        import obstore as obs

        if "zarr.json" in key:
            return get_zarr_metadata(self.xr_obj, key)
        manifest_index = parse_manifest_index(key)
        if manifest_index.variable == "__xarray_dataarray_variable__":
            path = self.xr_obj.data.manifest._paths[*manifest_index.indexes]
            offset = self.xr_obj.data.manifest._offsets[*manifest_index.indexes]
            length = self.xr_obj.data.manifest._lengths[*manifest_index.indexes]
            store_request = find_matching_store(stores=self.stores, request_key=path)
        else:
            path = self.xr_obj[manifest_index.variable].data.manifest._paths[
                *manifest_index.indexes
            ]
            offset = self.xr_obj[manifest_index.variable].data.manifest._offsets[
                *manifest_index.indexes
            ]
            length = self.xr_obj[manifest_index.variable].data.manifest._lengths[
                *manifest_index.indexes
            ]
            store_request = find_matching_store(stores=self.stores, request_key=path)
        chunk_end_exclusive = offset + length
        byte_range = _transform_byte_range(
            byte_range, chunk_start=offset, chunk_end_exclusive=chunk_end_exclusive
        )
        try:
            bytes = await obs.get_range_async(
                self.stores[store_request.store_id],
                store_request.key,
                start=byte_range.start,
                end=byte_range.end,
            )
            return prototype.buffer.from_bytes(bytes)  # type: ignore[arg-type]
        except _ALLOWED_EXCEPTIONS:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        raise NotImplementedError

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

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    @property
    def supports_partial_writes(self) -> bool:
        # docstring inherited
        return False

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

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        if isinstance(self.xr_obj, (DataArray, Dataset)):
            return list_dir_from_xr_obj(self.xr_obj, prefix)
        else:
            raise NotImplementedError(
                "Only DataArray and Datasets are currently supported"
            )


class _BoundedRequest(TypedDict):
    """Range request with a known start and end byte.

    These requests can be multiplexed natively on the Rust side with
    `obstore.get_ranges_async`.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    start: int
    """Start byte offset."""

    end: int
    """End byte offset."""


class _OtherRequest(TypedDict):
    """Offset or suffix range requests.

    These requests cannot be concurrent on the Rust side, and each need their own call
    to `obstore.get_async`, passing in the `range` parameter.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    path: str
    """The path to request from."""

    range: OffsetRange | SuffixRange | None
    """The range request type."""


class _Response(TypedDict):
    """A response buffer associated with the original index that it should be restored to."""

    original_request_index: int
    """The positional index in the original key_ranges input"""

    buffer: Buffer
    """The buffer returned from obstore's range request."""


async def _make_bounded_requests(
    store: _UpstreamObjectStore,
    path: str,
    requests: list[_BoundedRequest],
    prototype: BufferPrototype,
) -> list[_Response]:
    """Make all bounded requests for a specific file.

    `obstore.get_ranges_async` allows for making concurrent requests for multiple ranges
    within a single file, and will e.g. merge concurrent requests. This only uses one
    single Python coroutine.
    """
    import obstore as obs

    starts = [r["start"] for r in requests]
    ends = [r["end"] for r in requests]
    responses = await obs.get_ranges_async(store, path=path, starts=starts, ends=ends)

    buffer_responses: list[_Response] = []
    for request, response in zip(requests, responses, strict=True):
        buffer_responses.append(
            {
                "original_request_index": request["original_request_index"],
                "buffer": prototype.buffer.from_bytes(response),  # type: ignore[arg-type]
            }
        )

    return buffer_responses


async def _make_other_request(
    store: _UpstreamObjectStore,
    request: _OtherRequest,
    prototype: BufferPrototype,
) -> list[_Response]:
    """Make suffix or offset requests.

    We return a `list[_Response]` for symmetry with `_make_bounded_requests` so that all
    futures can be gathered together.
    """
    import obstore as obs

    if request["range"] is None:
        resp = await obs.get_async(store, request["path"])
    else:
        resp = await obs.get_async(
            store, request["path"], options={"range": request["range"]}
        )
    buffer = await resp.bytes_async()
    return [
        {
            "original_request_index": request["original_request_index"],
            "buffer": prototype.buffer.from_bytes(buffer),  # type: ignore[arg-type]
        }
    ]


async def _get_partial_values(
    store: _UpstreamObjectStore,
    prototype: BufferPrototype,
    key_ranges: Iterable[tuple[str, ByteRequest | None]],
) -> list[Buffer | None]:
    """Make multiple range requests.

    ObjectStore has a `get_ranges` method that will additionally merge nearby ranges,
    but it's _per_ file. So we need to split these key_ranges into **per-file** key
    ranges, and then reassemble the results in the original order.

    We separate into different requests:

    - One call to `obstore.get_ranges_async` **per target file**
    - One call to `obstore.get_async` for each other request.
    """
    key_ranges = list(key_ranges)
    per_file_bounded_requests: dict[str, list[_BoundedRequest]] = defaultdict(list)
    other_requests: list[_OtherRequest] = []

    for idx, (path, byte_range) in enumerate(key_ranges):
        if byte_range is None:
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": None,
                }
            )
        elif isinstance(byte_range, RangeByteRequest):
            per_file_bounded_requests[path].append(
                {
                    "original_request_index": idx,
                    "start": byte_range.start,
                    "end": byte_range.end,
                }
            )
        elif isinstance(byte_range, OffsetByteRequest):
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"offset": byte_range.offset},
                }
            )
        elif isinstance(byte_range, SuffixByteRequest):
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"suffix": byte_range.suffix},
                }
            )
        else:
            raise ValueError(f"Unsupported range input: {byte_range}")

    futs: list[Coroutine[Any, Any, list[_Response]]] = []
    for path, bounded_ranges in per_file_bounded_requests.items():
        futs.append(_make_bounded_requests(store, path, bounded_ranges, prototype))

    for request in other_requests:
        futs.append(_make_other_request(store, request, prototype))  # noqa: PERF401

    buffers: list[Buffer | None] = [None] * len(key_ranges)

    # TODO: this gather a list of list of Response; not sure if there's a way to
    # unpack these lists inside of an `asyncio.gather`?
    for responses in await asyncio.gather(*futs):
        for resp in responses:
            buffers[resp["original_request_index"]] = resp["buffer"]

    return buffers


def _transform_byte_range(byte_range, *, chunk_start, chunk_end_exclusive):
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
