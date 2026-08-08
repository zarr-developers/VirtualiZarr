from typing import Iterable

from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import ManifestStore
from virtualizarr.utils import soft_import

pydap = soft_import("pydap", "parsing dmrpp references", strict=False)


class DMRPPParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from a DMR++ file.

    Parameters
    ----------
    group
        The group within the file to be used as the Zarr root group for the ManifestStore.
    skip_variables
        Variables in the file that will be ignored when creating the ManifestStore.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given DMR++ file to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input DMR++ file (e.g., "s3://bucket/file.dmrpp").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the data source referenced by the DMR++ file.
        """

        from pydap.virtualizarr.parser import DMRParser

        parser = DMRParser(
            url=url,
            object_store=registry,
            skip_variables=self.skip_variables,
        )
        manifest_store = parser.parse_dataset(group=self.group)
        return manifest_store
