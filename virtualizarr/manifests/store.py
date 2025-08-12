from __future__ import annotations

import re
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias
from urllib.parse import urlparse

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.common import BytesLike

from virtualizarr.manifests.group import ManifestGroup
from virtualizarr.manifests.utils import construct_chunk_pattern
from virtualizarr.registry import ObjectStoreRegistry

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


def get_store_prefix(url: str) -> str:
    """
    Get a logical prefix to use for a url in an ObjectStoreRegistry
    """
    scheme, netloc, *_ = urlparse(url)
    return "" if scheme in {"", "file"} else f"{scheme}://{netloc}"


def parse_manifest_index(
    key: str, chunk_key_encoding: Literal[".", "/"] = "."
) -> tuple[int, ...]:
    """
    Extracts the chunk index from a `key` (a.k.a `node`) that represents a chunk of
    data in a Zarr hierarchy. The returned tuple can be used to index the ndarrays
    containing paths, offsets, and lengths in ManifestArrays.

    Parameters
    ----------
    key
        The key in the Zarr store to parse.
    chunk_key_encoding
        The chunk key separator used in the Zarr store.

    Returns
    -------
    tuple containing chunk indexes.

    Raises
    ------
    ValueError
        Raised if the key does not match the expected node structure for a chunk according the
        [Zarr V3 specification][https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/index.html].

    """
    # Keys ending in `/c` are scalar arrays. The paths, offsets, and lengths in a chunk manifest
    # of a scalar array should also be scalar arrays that can be indexed with an empty tuple.
    if key.endswith("/c"):
        return ()

    pattern = construct_chunk_pattern(chunk_key_encoding)
    # Expand pattern to include `/c` to protect against group structures that look like chunk structures
    pattern = rf"(?:^|/)c{chunk_key_encoding}{pattern}"
    # Look for f"/c{chunk_key_encoding"}" followed by digits and more /digits
    match = re.search(pattern, key)
    if not match:
        raise ValueError(
            f"Key {key} with chunk_key_encoding {chunk_key_encoding} did not match the expected pattern for nodes in the Zarr hierarchy."
        )
    chunk_component = (
        match.group().removeprefix("/").removeprefix(f"c{chunk_key_encoding}")
    )
    return tuple(int(ind) for ind in chunk_component.split(chunk_key_encoding))


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
        [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] that maps the URL scheme and netloc to  [ObjectStore][obstore.store.ObjectStore] instances,
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
        self, group: ManifestGroup, *, registry: ObjectStoreRegistry | None = None
    ) -> None:
        """Instantiate a new ManifestStore.

        Parameters
        ----------
        group
            [ManifestGroup][virtualizarr.manifests.ManifestGroup] containing Group metadata and mapping variable names to ManifestArrays
        registry
            A registry mapping the URL scheme and netloc to  [ObjectStore][obstore.store.ObjectStore] instances,
            allowing [ManifestStores][virtualizarr.manifests.ManifestStore] to read from different  [ObjectStore][obstore.store.ObjectStore] instances.
        """

        if not isinstance(group, ManifestGroup):
            raise TypeError

        super().__init__(read_only=True)
        self._registry = ObjectStoreRegistry() if registry is None else registry
        self._group = group

    def __str__(self) -> str:
        return f"ManifestStore(group={self._group}, registry={self._registry})"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # docstring inherited

        if key == "zarr.json":
            # Return group metadata
            return self._group.metadata.to_buffer_dict(
                prototype=default_buffer_prototype()
            )["zarr.json"]
        elif key.endswith("zarr.json"):
            # Return array metadata
            # TODO: Handle nested groups
            var, _ = key.split("/")
            return self._group.arrays[var].metadata.to_buffer_dict(
                prototype=default_buffer_prototype()
            )["zarr.json"]
        var = key.split("/")[0]
        marr = self._group.arrays[var]
        manifest = marr.manifest

        chunk_indexes = parse_manifest_index(
            key, marr.metadata.chunk_key_encoding.separator
        )

        path = manifest._paths[chunk_indexes]
        if path == "":
            return None
        offset = manifest._offsets[chunk_indexes]
        length = manifest._lengths[chunk_indexes]
        # Get the configured object store instance that matches the path
        store, path_after_prefix = self._registry.resolve(path)
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
        path_in_store = path_in_store.lstrip("/").removeprefix(prefix).lstrip("/")
        # Transform the input byte range to account for the chunk location in the file
        chunk_end_exclusive = offset + length
        byte_range = _transform_byte_range(
            byte_range, chunk_start=offset, chunk_end_exclusive=chunk_end_exclusive
        )

        # Actually get the bytes
        bytes = await store.get_range_async(
            path_in_store,
            start=byte_range.start,
            end=byte_range.end,
        )
        return prototype.buffer.from_bytes(bytes)  # type: ignore[arg-type]

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

        return construct_virtual_dataset(
            manifest_store=self,
            group=group,
            loadable_variables=loadable_variables,
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
