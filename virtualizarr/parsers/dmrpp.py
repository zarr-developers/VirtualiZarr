import io
import warnings
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from obspec_utils.protocols import ReadableStore
from obspec_utils.readers import EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from pydap.parsers.dmr import DMRPPParser as _DMRPPParser

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.utils import encode_cf_fill_value


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

        url = (
            url.removesuffix(".dap.dmrpp")
            if url.endswith(".dap.dmrpp")
            else url.removesuffix(".dmrpp")
        )

        parser = DMRParser(
            root=ET.parse(stream).getroot(),
            data_filepath=url,
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
        self.data_filepath = (
            data_filepath if data_filepath is not None else self.root.attrib["name"]
        )
        self.skip_variables = skip_variables or ()

    def dmrparser(self) -> _DMRPPParser:
        """Exposes the _DMRParser to external use (avoids breaking changes)"""
        return _DMRPPParser(
            root=self.root,
            data_filepath=self.data_filepath,
            skip_variables=self.skip_variables,
        )

    def parse_dataset(
        self,
        object_store: ReadableStore,
        group: str | None = None,
    ) -> ManifestStore:
        """
        Parses the given file and creates a ManifestStore.

        Parameters
        ----------
        group
            The group to parse. Ignored if no groups are present, and the entire
            dataset is parsed. If `None` or "/", and groups are present, the first group
            is parsed.  If not `None` or "/", and no groups are present, a UserWarning
            is issued indicating that the group will be ignored.

        Returns
        -------
        ManifestStore

        Examples
        --------
        Open a sample DMR++ file and parse the dataset
        """
        group = group or "/"
        ngroups = len(self.root.findall("dap:Group", self._NS))

        if ngroups == 0 and group != "/":
            warnings.warn(
                f"No groups in DMR++ file {self.data_filepath!r}; "
                f"ignoring group parameter {group!r}"
            )

        group_path = Path("/") if ngroups == 0 else Path("/") / group.removeprefix("/")

        dataset_element = self.dmrparser()._split_groups(self.root).get(group_path)

        if dataset_element is None:
            raise ValueError(
                f"Group {group_path} not found in DMR++ file {self.data_filepath!r}"
            )

        # get two dictionary containing relevant metadata
        vars_dict, attrs = self.dmrparser()._parse_dataset(dataset_element)

        manifest_dict: dict[str, ManifestArray] = {}

        for var in vars_dict.keys():
            chunkmanifest = ChunkManifest(vars_dict[var].pop("chunkmanifest", None))
            meta = dict(
                [
                    (k, v)
                    for k, v in vars_dict[var].items()
                    if k not in ["Maps", "fqn_dims"]
                ]
            )
            if "_FillValue" in meta["attributes"]:
                encoded_cf_fill_value = encode_cf_fill_value(
                    meta["attributes"]["_FillValue"], meta["data_type"]
                )
                meta["attributes"]["_FillValue"] = encoded_cf_fill_value
            metadata = create_v3_array_metadata(**meta)
            manifest_dict[var] = ManifestArray(
                metadata=metadata, chunkmanifest=chunkmanifest
            )
        manifest_group = ManifestGroup(arrays=manifest_dict, attributes=attrs)
        registry = ObjectStoreRegistry()
        registry.register(self.data_filepath, object_store)

        return ManifestStore(registry=registry, group=manifest_group)
