from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from obspec_utils.protocols import ReadableFile, ReadableStore
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import ManifestStore

# Type alias for reader factories
ReaderFactory = Callable[[ReadableStore, str], ReadableFile]


@runtime_checkable
class Parser(Protocol):
    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the contents of a given data source to produce a ManifestStore.

        Effectively maps the contents of the data source (including the metadata, compression codecs, chunk byte offsets)
        to the Zarr data model.

        Parameters
        ----------
        url
            The URL of the input data source (e.g., "s3://bucket/file.nc").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a Zarr representation of the parsed data source.
        """
        ...
