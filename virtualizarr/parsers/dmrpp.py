import io
from typing import Iterable
from xml.etree import ElementTree as ET

from obspec_utils.readers import EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ManifestStore,
)


class DMRPPParser:
    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the __call__ method.

        Parameters
        ----------
        group
            The group within the file to be used as the Zarr root group for the ManifestStore.
        skip_variables
            Variables in the file that will be ignored when creating the ManifestStore.
        """

        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given DMR++ file to product a
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

        parser = DMRParser(
            root=ET.parse(stream).getroot(),
            data_filepath=url.removesuffix(".dmrpp"),
            skip_variables=self.skip_variables,
        )
        manifest_store = parser.parse_dataset(object_store=store, group=self.group)
        return manifest_store


class DMRParser:
    """
    Parser for the OPeNDAP DMR++ XML format.
    Reads groups, dimensions, coordinates, data variables, encoding, chunk manifests, and attributes.
    Highly modular to allow support for older dmrpp schema versions. Includes many utility functions to extract
    different information such as finding all variable tags, splitting hdf5 groups, parsing dimensions, and more.

    OPeNDAP DMR++ homepage: https://docs.opendap.org/index.php/DMR%2B%2B
    """

    # DAP and DMRPP XML namespaces
    _NS = {
        "dap": "http://xml.opendap.org/ns/DAP/4.0#",
        "dmrpp": "http://xml.opendap.org/dap/dmrpp/1.0.0#",
    }

    root: ET.Element
    data_filepath: str

    def __init__(
        self,
        root: ET.Element,
        data_filepath: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        """
        Initialize the DMRParser with the given DMR++ file contents and source data file path.

        Parameters
        ----------
        root
            Root of the xml tree structure of a DMR++ file.
        data_filepath
            The path to the actual data file that will be set in the chunk manifests.
            If None, the data file path is taken from the DMR++ file.
        """
        self.root = root
        self._validation_issues: list[str] = []
        self.data_filepath = data_filepath
        self.skip_variables = skip_variables or ()
