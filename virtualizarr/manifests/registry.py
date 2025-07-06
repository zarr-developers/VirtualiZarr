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
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(
            f"Urls are expected to contain a scheme, received {url} which parsed to {parsed}"
        )
    return UrlKey(parsed.scheme, parsed.netloc)


class PathPart:
    """Represents a single path segment"""

    @staticmethod
    def parse(segment: str) -> "PathPart":
        """Parse a path segment, validating it's acceptable"""
        if not segment or "/" in segment:
            raise ValueError(f"Invalid path segment: {segment}")
        return PathPart(segment)

    def __init__(self, segment: str):
        self.segment = segment

    def __str__(self) -> str:
        return self.segment


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
    def __init__(self) -> None:
        # Mapping from UrlKey (containing scheme and netlocs) to PathEntry
        self.map: Dict[UrlKey, PathEntry] = {}

    def register(self, url: Url, store: ObjectStore) -> Optional[ObjectStore]:
        """Register a store for the given URL"""
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

        # Replace the store and return the old one
        old_store = entry.store
        entry.store = store
        return old_store

    def resolve(self, url: Url) -> Tuple[ObjectStore, Path]:
        """Resolve a URL to an ObjectStore and path"""
        parsed = urlparse(url)

        key = UrlKey(parsed.scheme, parsed.netloc)

        if key in self.map:
            result = self.map[key].lookup(parsed.path)
            if result:
                store, depth = result
                path = "/".join(list(path_segments(parsed.path))[depth:])
                return store, path

        raise ValueError(f"Could not find an ObjectStore matching the url {url}")


def path_segments(path: str) -> Iterator[str]:
    """
    Returns the non-empty segments of a path

    Note: We filter out empty segments unlike urllib.parse
    """
    return filter(lambda x: x, path.split("/"))
