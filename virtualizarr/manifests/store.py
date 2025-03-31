from __future__ import annotations

import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import urlparse

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

from virtualizarr.manifests.array import ManifestArray
from virtualizarr.manifests.group import ManifestGroup

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from typing import Any

    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import BytesLike


__all__ = ["ManifestStore"]


_ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from zarr.core.buffer import default_buffer_prototype

from virtualizarr.vendor.zarr.metadata import dict_to_buffer

if TYPE_CHECKING:
    from obstore.store import ObjectStore  # type: ignore[import-not-found]

    StoreDict: TypeAlias = dict[str, ObjectStore]


@dataclass
class StoreRequest:
    """Dataclass for matching a key to the store instance"""

    store: ObjectStore
    """The ObjectStore instance to use for making the request."""
    key: str
    """The key within the store to request."""


async def list_dir_from_manifest_arrays(
    arrays: Mapping[str, ManifestArray], prefix: str
) -> AsyncGenerator[str]:
    """Create the expected results for Zarr's `store.list_dir()` from an Xarray DataArrray or Dataset

    Parameters
    ----------
    arrays : Mapping[str, ManifestArrays]
    prefix : str

    Returns
    -------
    AsyncIterator[str]
    """
    # TODO shouldn't this just accept a ManifestGroup instead?
    # Start with expected group level metadata
    raise NotImplementedError


def get_zarr_metadata(manifest_group: ManifestGroup, key: str) -> Buffer:
    """
    Generate the expected Zarr V3 metadata from a virtual dataset.

    Group metadata is returned for all Datasets and Array metadata
    is returned for all DataArrays.

    Combines the ManifestArray metadata with the attrs from the DataArray
    and adds `dimension_names` for all arrays if not already provided.

    Parameters
    ----------
    manifest_group : ManifestGroup
    key : str

    Returns
    -------
    Buffer
    """
    # If requesting the root metadata, return the standard group metadata with additional dataset specific attributes

    if key == "zarr.json":
        metadata = manifest_group.metadata.to_dict()
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())
    else:
        var, _ = key.split("/")
        metadata = manifest_group._arrays[var].metadata.to_dict()
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())


def parse_manifest_index(
    key: str, chunk_key_encoding: str = "."
) -> tuple[str, tuple[int, ...]]:
    """
    Splits `key` provided to a zarr store into the variable indicated
    by the first part and the chunk index from the 3rd through last parts,
    which can be used to index into the ndarrays containing paths, offsets,
    and lengths in ManifestArrays.

    Currently only works for 1d+ arrays with a tree depth of one from the
    root Zarr group.

    Parameters
    ----------
    key : str
    chunk_key_encoding : str

    Returns
    -------
    ManifestIndex
    """
    parts = key.split("/")
    var = parts[0]
    # Assume "c" is the second part
    # TODO: Handle scalar array case with "c" holds the data
    if chunk_key_encoding == "/":
        indexes = tuple(int(ind) for ind in parts[2:])
    else:
        indexes = tuple(int(ind) for ind in parts[2].split(chunk_key_encoding))
    return var, indexes


def find_matching_store(stores: StoreDict, request_key: str) -> StoreRequest:
    """
    Find the matching store based on the store keys and the beginning of the URI strings,
    to fetch data from the appropriately configured ObjectStore.

    Parameters:
    -----------
    stores : StoreDict
        A dictionary with URI prefixes for different stores as keys
    request_key : str
        A string to match against the dictionary keys

    Returns:
    --------
    StoreRequest
    """
    # Sort keys by length in descending order to ensure longer, more specific matches take precedence
    sorted_keys = sorted(stores.keys(), key=len, reverse=True)

    # Check each key to see if it's a prefix of the uri_string
    for key in sorted_keys:
        if request_key.startswith(key):
            parsed_key = urlparse(request_key)
            return StoreRequest(store=stores[key], key=parsed_key.path)
    # if no match is found, raise an error
    raise ValueError(
        f"Expected the one of stores.keys() to match the data prefix, got {stores.keys()} and {request_key}"
    )


class ManifestStore(Store):
    """A read-only Zarr store that uses obstore to access data on AWS, GCP, Azure. The requests
    from the Zarr API are redirected using the :class:`virtualizarr.manifests.ManifestGroup` containing
    multiple :class:`virtualizarr.manifests.ManifestArray`,
    allowing for virtually interfacing with underlying data in other file format.


    Parameters
    ----------
    manifest_group : ManifestGroup
        Manifest Group containing Group metadata and mapping variable names to ManifestArrays
    stores : dict[prefix, :class:`obstore.store.ObjectStore`]
        A mapping of url prefixes to obstore Store instances set up with the proper credentials.

        The prefixes are matched to the URIs in the ManifestArrays to determine which store to
        use for making requests.

    Warnings
    --------
    ManifestStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.

    Notes
    -----
    Modified from https://github.com/zarr-developers/zarr-python/pull/1661
    """

    _manifest_group: ManifestGroup
    _stores: StoreDict

    def __eq__(self, value: object):
        NotImplementedError

    def __init__(
        self,
        manifest_group: ManifestGroup,
        *,
        stores: StoreDict,  # TODO: Consider using a sequence of tuples rather than a dict (see https://github.com/zarr-developers/VirtualiZarr/pull/490#discussion_r2010717898).
    ) -> None:
        """Instantiate a new ManifestStore

        Parameters
        ----------
        manifest_group : ManifestGroup
            Manifest Group containing Group metadata and mapping variable names to ManifestArrays
        stores : dict[prefix, :class:`obstore.store.ObjectStore`]
            A mapping of url prefixes to obstore Store instances set up with the proper credentials.

            The prefixes are matched to the URIs in the ManifestArrays to determine which store to
            use for making requests.
        """
        for store in stores.values():
            if not store.__class__.__module__.startswith("obstore"):
                raise TypeError(f"expected ObjectStore class, got {store!r}")
        # TODO: Don't allow stores with prefix
        # TODO: Type check the manifest arrays
        super().__init__(read_only=True)
        self._stores = stores
        self._manifest_group = manifest_group

    def __str__(self) -> str:
        return f"ManifestStore({self._manifest_group}, {self._stores})"

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        stores = state["_stores"].copy()
        for k, v in stores.items():
            stores[k] = pickle.dumps(v)
        state["_stores"] = stores
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        stores = state["_stores"].copy()
        for k, v in stores.items():
            stores[k] = pickle.loads(v)
        state["_stores"] = stores
        self.__dict__.update(state)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited
        import obstore as obs

        if key.endswith("zarr.json"):
            return get_zarr_metadata(self._manifest_group, key)
        var, chunk_key = parse_manifest_index(key)
        marr = self._manifest_group._arrays[var]
        manifest = marr._manifest

        path = manifest._paths[*chunk_key]
        offset = manifest._offsets[*chunk_key]
        length = manifest._lengths[*chunk_key]
        # Get the  configured object store instance that matches the path
        store_request = find_matching_store(stores=self._stores, request_key=path)
        # Transform the input byte range to account for the chunk location in the file
        chunk_end_exclusive = offset + length
        byte_range = _transform_byte_range(
            byte_range, chunk_start=offset, chunk_end_exclusive=chunk_end_exclusive
        )
        # Actually get the bytes
        try:
            bytes = await obs.get_range_async(
                store_request.store,
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
        # TODO: Implement using private functions from the upstream Zarr obstore integration
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

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        yield "zarr.json"
        for k in self._manifest_group._arrays.keys():
            yield k


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
