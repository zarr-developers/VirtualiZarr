import io
from typing import Iterable
from xml.etree import ElementTree as ET

# from obspec_utils.protocols import ReadableStore
from obspec_utils.readers import EagerStoreReader
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
        store, path_in_store = registry.resolve(url)
        reader = EagerStoreReader(store=store, path=path_in_store)
        file_bytes = reader.readall()
        stream = io.BytesIO(file_bytes)

        url = (
            url.removesuffix(".dap.dmrpp")
            if url.endswith(".dap.dmrpp")
            else url.removesuffix(".dmrpp")
        )

        from pydap.virtualizarr.parser import DMRParser

        parser = DMRParser(
            root=ET.parse(stream).getroot(),
            data_filepath=url,
            skip_variables=self.skip_variables,
        )
        manifest_store = parser.parse_dataset(object_store=store, group=self.group)
        return manifest_store
