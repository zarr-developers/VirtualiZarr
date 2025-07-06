from __future__ import annotations

from typing import Protocol, runtime_checkable

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.registry import ObjectStoreRegistry


@runtime_checkable
class Parser(Protocol):
    def __call__(
        self,
        file_url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore: ...

    """
    Parse the contents of a given file to produce a ManifestStore.

    Effectively maps the contents of the file (e.g. metadata, compression codecs, chunk byte offsets) to the Zarr data model.

    Parameters
    ----------
    file_url
        The URL of the input file (e.g., "s3://bucket/file.nc").
    registry
        An [ObjectStoreRegistry][virtualizarr.manifests.ObjectStoreRegistry] for resolving urls and reading data.

    Returns
    -------
    ManifestStore
        A ManifestStore which provides a Zarr representation of the parsed file.
    """
