"""
Based on https://docs.rs/object_store/0.12.2/src/object_store/registry.rs.html#176-218

This module may eventually be upstreamed into https://github.com/developmentseed/obstore
"""

from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple, TypeAlias
from urllib.parse import urlparse

if TYPE_CHECKING:
    from obstore.store import (
        ObjectStore,
    )

Url: TypeAlias = str
Path: TypeAlias = str

UrlKey = namedtuple("UrlKey", ["scheme", "netloc"])


def get_url_key(url: Url) -> UrlKey:
    """
    Generate the UrlKey containing a url's scheme and authority/netloc that is used a the
    primary key's in a [ObjectStoreRegistry.map][virtualizarr.registry.ObjectStoreRegistry.map]

    Parameters
    ----------
    url
        Url to generate a UrlKey from

    Returns
    -------
        NamedTuple containing the Url's scheme and authority/netloc

    Raises
    ------
    ValueError
        If provided Url does not contain a scheme based on [urllib.parse.urlparse][]
    """
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(
            f"Urls are expected to contain a scheme (e.g., `file://` or `s3://`), received {url} which parsed to {parsed}"
        )
    return UrlKey(parsed.scheme, parsed.netloc)


class PathEntry:
    """
    Construct a tree of path segments starting from the root

    For example the following paths:
    * `/` => store1
    * `/foo/bar` => store2

    Would be represented by:
    store: Some(store1)
    children:
      foo:
        store: None
        children:
          bar:
            store: Some(store2)
    """

    def __init__(self) -> None:
        self.store: Optional[ObjectStore] = None
        self.children: Dict[str, "PathEntry"] = {}

    def lookup(self, to_resolve: str) -> Optional[Tuple[ObjectStore, int]]:
        """
        Lookup a store based on URL path

        Returns the store and its path segment depth
        """
        current = self
        ret = (self.store, 0) if self.store is not None else None
        depth = 0

        # Traverse the PathEntry tree to find the longest match
        for segment in path_segments(to_resolve):
            if segment in current.children:
                current = current.children[segment]
                depth += 1
                if current.store is not None:
                    ret = (current.store, depth)
            else:
                break

        return ret


class ObjectStoreRegistry:
    def __init__(self, stores: dict[Url, ObjectStore] | None = None) -> None:
        """
        Create a new store registry that matches the provided Urls and
        [ObjectStore][obstore.store.ObjectStore] instances.


        Parameters
        ----------
        stores
            Mapping of [Url][virtualizarr.registry.Url] to the [ObjectStore][obstore.store.ObjectStore]
            to be registered under the [Url][virtualizarr.registry.Url].

        Examples
        --------

        ```python exec="on" source="above" session="registry-examples"
        from obstore.store import S3Store
        from virtualizarr.registry import ObjectStoreRegistry

        s3store = S3Store(bucket="my-bucket-1", prefix="orig-path")
        reg = ObjectStoreRegistry({"s3://my-bucket-1": s3store})

        ret, path = reg.resolve("s3://my-bucket-1/orig-path/group/my-file.nc")
        assert path == "group/my-file.nc"
        assert ret is s3store
        ```
        """
        # Mapping from UrlKey (containing scheme and netlocs) to PathEntry
        self.map: Dict[UrlKey, PathEntry] = {}
        stores = stores or {}
        for url, store in stores.items():
            self.register(url, store)

    def register(self, url: Url, store: ObjectStore) -> None:
        """
        Register a new store for the provided store [Url][virtualizarr.registry.Url].

        If a store with the same [Url][virtualizarr.registry.Url]  existed before, it is replaced.

        Parameters
        ----------
        url
            [Url][virtualizarr.registry.Url] to registry the [ObjectStore][obstore.store.ObjectStore] under.
        store
            [ObjectStore][obstore.store.ObjectStore] instance to register using the
            provided [Url][virtualizarr.registry.Url].

        Examples
        --------

        ```python exec="on" source="above" session="registry-examples"
        from obstore.store import S3Store
        from virtualizarr.registry import ObjectStoreRegistry

        reg = ObjectStoreRegistry()
        orig_store = S3Store(bucket="my-bucket-1", prefix="orig-path")
        reg.register("s3://my-bucket-1", orig_store)

        new_store = S3Store(bucket="my-bucket-1", prefix="updated-path")
        reg.register("s3://my-bucket-1", new_store)
        ```
        """
        parsed = urlparse(url)

        key = get_url_key(url)

        if key not in self.map:
            self.map[key] = PathEntry()

        entry = self.map[key]

        # Navigate to the correct path in the tree
        for segment in path_segments(parsed.path):
            if segment not in entry.children:
                entry.children[segment] = PathEntry()
            entry = entry.children[segment]
        # Update the store
        entry.store = store

    def resolve(self, url: Url) -> Tuple[ObjectStore, Path]:
        """
        Resolve an URL within the [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry].

        If [ObjectStoreRegistry.register][virtualizarr.registry.ObjectStoreRegistry.register] has been called
        with a URL with the same scheme and authority/netloc as the object URL, and a path that is a prefix
        of the provided url's, it is returned along with the trailing path. Paths are matched on a
        path segment basis, and in the event of multiple possibilities the longest path match is used.

        Parameters
        ----------
        url
            Url to resolve in the [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry]

        Returns
        -------
        ObjectStore
            The [ObjectStore][obstore.store.ObjectStore] stored at the resolved url.
        Path
            The trailing portion of the url after the prefix of the matching store in the
            [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry].

        Raises
        ------
        ValueError
            If the URL cannot be resolved, meaning that [ObjectStoreRegistry.register][virtualizarr.registry.ObjectStoreRegistry.register]
            has not been called with a URL with the same scheme and authority/netloc as the object URL, and a path that is a prefix
            of the provided url's.

        Examples
        --------

        ```python exec="on" source="above" session="registry-resolve-examples"
        from obstore.store import MemoryStore, S3Store
        from virtualizarr.registry import ObjectStoreRegistry

        registry = ObjectStoreRegistry()
        memstore1 = MemoryStore()
        registry.register("s3://bucket1", memstore1)
        url = "s3://bucket1/path/to/object"
        ret, path = registry.resolve(url)
        assert path == "path/to/object"
        assert ret is memstore1
        print(f"Resolved url: `{url}` to store: `{ret}` and path: `{path}`")
        ```

        ```python exec="on" source="above" session="registry-resolve-examples"
        memstore2 = MemoryStore()
        base = "https://s3.region.amazonaws.com/bucket"
        registry.register(base, memstore2)

        url = "https://s3.region.amazonaws.com/bucket/path/to/object"
        ret, path = registry.resolve(url)
        assert path == "bucket/path/to/object"
        assert ret is memstore2
        print(f"Resolved url: `{url}` to store: `{ret}` and path: `{path}`")
        ```

        ```python exec="on" source="above" session="registry-resolve-examples"
        s3store = S3Store(bucket = "my-bucket", prefix="my-data/prefix/")
        registry.register("s3://my-bucket", s3store)
        ret, path = registry.resolve("s3://my-bucket/my-data/prefix/my-file.nc")
        assert path == "my-file.nc"
        assert ret is s3store
        ```
        """
        parsed = urlparse(url)
        path = parsed.path

        key = UrlKey(parsed.scheme, parsed.netloc)

        if key in self.map:
            result = self.map[key].lookup(path)
            if result:
                store, _ = result
                if hasattr(store, "prefix") and store.prefix:
                    prefix = str(store.prefix).lstrip("/")
                    path_after_prefix = (
                        path.lstrip("/").removeprefix(prefix).lstrip("/")
                    )
                elif hasattr(store, "url"):
                    prefix = urlparse(store.url).path.lstrip("/")
                    path_after_prefix = (
                        path.lstrip("/").removeprefix(prefix).lstrip("/")
                    )
                else:
                    path_after_prefix = path.lstrip("/")
                return store, path_after_prefix
        raise ValueError(f"Could not find an ObjectStore matching the url `{url}`")


def path_segments(path: str) -> Iterator[str]:
    """
    Returns the non-empty segments of a path

    Note: We filter out empty segments unlike urllib.parse
    """
    return filter(lambda x: x, path.split("/"))


__all__ = ["ObjectStoreRegistry"]
