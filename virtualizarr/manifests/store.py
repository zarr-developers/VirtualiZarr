from __future__ import annotations

import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING
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
    from collections.abc import AsyncGenerator, Iterable, Mapping
    from typing import Any

    import xarray as xr
    from obstore.store import (
        ObjectStore,  # type: ignore[import-not-found]
        S3Config,
    )
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
from typing import TYPE_CHECKING, Any

from zarr.core.buffer import default_buffer_prototype

from virtualizarr.vendor.zarr.metadata import dict_to_buffer


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
        metadata = manifest_group.arrays[var].metadata.to_dict()
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())


def parse_manifest_index(key: str, chunk_key_encoding: str = ".") -> tuple[int, ...]:
    """
    Splits `key` provided to a zarr store into the variable indicated
    by the first part and the chunk index from the 3rd through last parts,
    which can be used to index into the ndarrays containing paths, offsets,
    and lengths in ManifestArrays.

    Parameters
    ----------
    key : str
    chunk_key_encoding : str

    Returns
    -------
    tuple containing chunk indexes
    """
    if key.endswith("c"):
        # Scalar arrays hold the data in the "c" key
        raise NotImplementedError(
            "Scalar arrays are not yet supported by ManifestStore"
        )
    parts = key.split(
        "c/"
    )  # TODO: Open an issue upstream about the Zarr spec indicating this should be f"c{chunk_key_encoding}" rather than always "c/"
    return tuple(int(ind) for ind in parts[1].split(chunk_key_encoding))


def _default_object_store(
    filepath: str, config: S3Config | None = None
) -> tuple[str, ObjectStore]:
    import obstore as obs

    parsed = urlparse(filepath)

    if parsed.scheme in ["", "file"]:
        return "file://", obs.store.LocalStore()
    if parsed.scheme == "s3":
        config = config or {"skip_signature": True}
        config["virtual_hosted_style_request"] = False
        bucket = parsed.netloc
        return f"s3://{bucket}", obs.store.S3Store(
            bucket,
            client_options={"allow_http": True},
            config=config,
        )

    raise NotImplementedError(f"{parsed.scheme} is not yet supported")


def _sort_stores_by_prefix_length(input_dict):
    sorted_items = sorted(input_dict.items(), key=lambda x: len(x[0]), reverse=True)
    return dict(sorted_items)


class ObjectStoreRegistry:
    """
    ObjectStoreRegistry maps URLs to ObjectStore instances, and allows ManifestStores to read from different ObjectStore instances.
    """

    _stores: dict[str, ObjectStore]

    @classmethod
    def __init__(self, stores: dict | None = None):
        stores = stores or {}
        for store in stores.values():
            if not store.__class__.__module__.startswith("obstore"):
                raise TypeError(f"expected ObjectStore class, got {store!r}")
        self._stores = _sort_stores_by_prefix_length(stores)

    def register_store(self, url: str, store: ObjectStore):
        """
        Register a store using the given url

        If a store with the same key existed before, it is replaced
        """
        self._stores[url] = store
        self._stores = _sort_stores_by_prefix_length(self._stores)

    def get_store(self, url: str) -> ObjectStore:
        """
        Get a suitable store for the provided URL. For example:

            - URL with scheme file:/// or no scheme will return the default LocalFS store
            - URL with scheme s3://bucket/ will return the S3 store

        If no `ObjectStore` is found for the `url`, ad-hoc discovery may be executed depending on the
        `url`. An `ObjectStore` may be lazily created and registered.

        Parameters:
        -----------
        url : str
            A url to identify the appropriate object_store instance

        Returns:
        --------
        StoreRequest
        """
        # Check each key to see if it's a prefix of the uri_string
        for prefix, store in self._stores.items():
            if url.startswith(prefix):
                # Return an existing configured store and parsed request path
                return store
        # Use anonymous default store if not in pre-configured stores
        prefix, store = _default_object_store(url)
        # Register for future use
        self.register_store(prefix, store)
        # Return the new store and and parsed request path
        return store


class ManifestStore(Store):
    """
    A read-only Zarr store that uses obstore to access data on AWS, GCP, Azure. The requests
    from the Zarr API are redirected using the :class:`virtualizarr.manifests.ManifestGroup` containing
    multiple :class:`virtualizarr.manifests.ManifestArray`,
    allowing for virtually interfacing with underlying data in other file format.

    Parameters
    ----------
    group : ManifestGroup
        Root group of the store.
        Contains group metadata, ManifestArrays, and any subgroups.
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

    _group: ManifestGroup
    _store_registry: ObjectStoreRegistry

    def __eq__(self, value: object):
        NotImplementedError

    def __init__(
        self, group: ManifestGroup, *, store_registry: ObjectStoreRegistry | None = None
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

        # TODO: Don't allow stores with prefix
        if not isinstance(group, ManifestGroup):
            raise TypeError

        super().__init__(read_only=True)
        if store_registry is None:
            store_registry = ObjectStoreRegistry()
        self._store_registry = store_registry
        self._group = group

    def __str__(self) -> str:
        return f"ManifestStore(group={self._group}, stores={self._store_registry})"

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        stores = state["_store_registry"]._stores.copy()
        for k, v in stores.items():
            stores[k] = pickle.dumps(v)
        state["_store_registry"] = stores
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        stores = state["_store_registry"].copy()
        for k, v in stores.items():
            stores[k] = pickle.loads(v)
        state["_store_registry"] = ObjectStoreRegistry(stores)
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
            return get_zarr_metadata(self._group, key)
        var = key.split("/")[0]
        marr = self._group.arrays[var]
        manifest = marr.manifest

        chunk_indexes = parse_manifest_index(
            key, marr.metadata.chunk_key_encoding.separator
        )
        path = manifest._paths[*chunk_indexes]
        offset = manifest._offsets[*chunk_indexes]
        length = manifest._lengths[*chunk_indexes]
        # Get the configured object store instance that matches the path
        store = self._store_registry.get_store(path)
        # Truncate path to match Obstore expectations
        key = urlparse(path).path
        # Transform the input byte range to account for the chunk location in the file
        chunk_end_exclusive = offset + length
        byte_range = _transform_byte_range(
            byte_range, chunk_start=offset, chunk_end_exclusive=chunk_end_exclusive
        )
        # Actually get the bytes
        try:
            bytes = await obs.get_range_async(
                store,
                key,
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
        for k in self._group.arrays.keys():
            yield k

    def to_virtual_dataset(
        self,
        group="",
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, xr.Index] | None = None,
    ) -> "xr.Dataset":
        """
        Create a "virtual" xarray dataset containing the contents of one zarr group.

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

        return construct_virtual_dataset(
            manifest_store=self,
            group=group,
            loadable_variables=loadable_variables,
            indexes=indexes,
            decode_times=decode_times,
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
