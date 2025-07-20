from __future__ import annotations

from typing import Protocol, runtime_checkable

from virtualizarr.manifests import ManifestStore
from virtualizarr.registry import ObjectStoreRegistry


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
            An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a Zarr representation of the parsed data source.
        """
        ...
