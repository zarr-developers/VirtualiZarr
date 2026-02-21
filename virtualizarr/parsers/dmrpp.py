import io
import warnings
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np
from obspec_utils.protocols import ReadableStore
from obspec_utils.readers import EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.utils import encode_cf_fill_value
from virtualizarr.types import ChunkKey


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
    # DAP data types to numpy data types
    _DAP_NP_DTYPE = {
        "Byte": "uint8",
        "UByte": "uint8",
        "Int8": "int8",
        "UInt8": "uint8",
        "Int16": "int16",
        "UInt16": "uint16",
        "Int32": "int32",
        "UInt32": "uint32",
        "Int64": "int64",
        "UInt64": "uint64",
        "Url": "object",
        "Float32": "float32",
        "Float64": "float64",
        "String": "object",
    }
    # Default zlib compression value
    _DEFAULT_ZLIB_VALUE = 6
    # Encoding keys that should be removed from attributes and placed in xarray encoding dict
    # _ENCODING_KEYS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}
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
        data_filepath_from_root = self._get_attrib(self.root, "name", required=True)
        assert data_filepath_from_root is not None  # required=True guarantees non-None
        self.data_filepath = (
            data_filepath if data_filepath is not None else data_filepath_from_root
        )
        self.skip_variables = skip_variables or ()

    def _get_attrib(
        self, element: ET.Element, attrib_name: str, required: bool = False
    ) -> str | None:
        """
        Safely get an attribute from an XML element, logging validation issues.

        Parameters
        ----------
        element
            The XML element to get the attribute from.
        attrib_name
            The name of the attribute to get.
        required
            If True, raises a ValueError when the attribute is missing. If False,
            returns None and logs the issue.

        Returns
        -------
        str | None
            The attribute value if found, None otherwise.

        Raises
        ------
        ValueError
            If required is True and the attribute is not found.
        """
        if attrib_name in element.attrib:
            return element.attrib[attrib_name]

        element_info = (
            element.tag
            if "name" not in element.attrib
            else f"{element.tag}[@name='{element.attrib['name']}']"
        )
        issue_msg = (
            f"Missing required attribute '{attrib_name}' in element: {element_info}"
        )
        self._validation_issues.append(issue_msg)

        if required:
            raise ValueError(issue_msg)

        return None

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
        dataset_element = self._split_groups(self.root).get(group_path)

        if dataset_element is None:
            raise ValueError(
                f"Group {group_path} not found in DMR++ file {self.data_filepath!r}"
            )

        manifest_group = self._parse_dataset(dataset_element)
        registry: ObjectStoreRegistry = ObjectStoreRegistry()
        registry.register(self.data_filepath, object_store)

        return ManifestStore(registry=registry, group=manifest_group)

    def find_node_fqn(self, fqn: str) -> ET.Element:
        """
        Find the element in the root element by converting the fully qualified name to an xpath query.

        E.g. fqn = "/a/b" --> root.find("./*[@name='a']/*[@name='b']")

        See more about OPeNDAP fully qualified names (FQN) here: https://docs.opendap.org/index.php/DAP4:_Specification_Volume_1#Fully_Qualified_Names

        Parameters
        ----------
        fqn
            The fully qualified name of an element. For example, "/a/b".

        Returns
        -------
        ET.Element
            The matching node found within the root element.

        Raises
        ------
        ValueError
            If the fully qualified name is not found in the root element.
        """
        if fqn == "/":
            return self.root

        elements = fqn.strip("/").split("/")  # /a/b/ --> ['a', 'b']
        xpath_segments = [f"*[@name='{element}']" for element in elements]
        xpath_query = "/".join([".", *xpath_segments])  # "./[*[@name='a']/*[@name='b']"

        if (element := self.root.find(xpath_query, self._NS)) is None:
            raise ValueError(f"Path {fqn} not found in provided root")

        return element

    def _split_groups(self, root: ET.Element) -> dict[Path, ET.Element]:
        """
        Split the input <Dataset> element into several <Dataset> ET.Elements by <Group> name.
        E.g. {"/": <Dataset>, "left": <Dataset>, "right": <Dataset>}

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR file.

        Returns
        -------
        dict[Path, ET.Element]
        """
        all_groups: dict[Path, ET.Element] = {}
        dataset_tags = [
            d for d in root if d.tag != "{" + self._NS["dap"] + "}" + "Group"
        ]
        if len(dataset_tags) > 0:
            all_groups[Path("/")] = ET.Element(root.tag, root.attrib)
            all_groups[Path("/")].extend(dataset_tags)
        all_groups.update(self._split_groups_recursive(root, Path("/")))
        return all_groups

    def _split_groups_recursive(
        self, root: ET.Element, current_path=Path("")
    ) -> dict[Path, ET.Element]:
        group_dict: dict[Path, ET.Element] = {}
        for g in root.iterfind("dap:Group", self._NS):
            group_name = self._get_attrib(g, "name", required=True)
            if group_name is None:
                continue
            new_path = current_path / Path(group_name)
            dataset_tags = [
                d for d in g if d.tag != "{" + self._NS["dap"] + "}" + "Group"
            ]
            group_dict[new_path] = ET.Element(g.tag, g.attrib)
            group_dict[new_path].extend(dataset_tags)
            group_dict.update(self._split_groups_recursive(g, new_path))
        return group_dict

    def _parse_dataset(
        self,
        root: ET.Element,
    ) -> ManifestGroup:
        """
        Parse the dataset using the root element of the DMR++ file.

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR++ file.

        Returns
        -------
        ManifestGroup
        """

        manifest_dict: dict[str, ManifestArray] = {}
        for var_tag in self._find_var_tags(root):
            var_name = self._get_attrib(var_tag, "name")
            if var_name and var_name not in self.skip_variables:
                try:
                    variable = self._parse_variable(var_tag)
                    manifest_dict[var_name] = variable
                except (UnboundLocalError, ValueError):
                    warnings.warn(
                        f"This DMRpp contains the variable {var_name} that could not"
                        " be parsed. Consider adding it to the list  of skipped "
                        "variables, or opening an issue to help resolve this"
                    )

        # Attributes
        attrs: dict[str, str] = {}
        # Look for an attribute tag called "HDF5_GLOBAL" and unpack it
        hdf5_global_attrs = root.find("dap:Attribute[@name='HDF5_GLOBAL']", self._NS)
        if hdf5_global_attrs is not None:
            # Remove the container attribute and add its children to the root dataset
            root.remove(hdf5_global_attrs)
            root.extend(hdf5_global_attrs)
        for attr_tag in root.iterfind("dap:Attribute", self._NS):
            attrs.update(self._parse_attribute(attr_tag))

        return ManifestGroup(
            arrays=manifest_dict,
            attributes=attrs,
        )

    def _find_var_tags(self, root: ET.Element) -> list[ET.Element]:
        """
        Find all variable tags in the DMR++ file. Also known as array tags.
        Tags are labeled with the DAP data type. E.g. <Float32>, <Int16>, <String>

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR++ file.

        Returns
        -------
        list[ET.Element]
        """
        vars_tags: list[ET.Element] = []
        for dap_dtype in self._DAP_NP_DTYPE:
            vars_tags += root.findall(f"dap:{dap_dtype}", self._NS)
        return vars_tags

    def _parse_dim(self, root: ET.Element, dim_index: int = 0) -> dict[str, int]:
        """
        Parse single <Dim> or <Dimension> tag

        If the tag has no name attribute, it is a phony dimension. E.g. <Dim size="300"/> --> {"phony_dim_0": 300}
        If the tag has both name and size attributes, it is a regular dimension. E.g. <Dim name="lat" size="1447"/> --> {"lat": 1447}

        Parameters
        ----------
        root : ET.Element
            The root element Dim/Dimension tag
        dim_index : int
            Index of the dimension, used for naming phony dimensions

        Returns
        -------
        dict
            E.g. {"time": 1, "lat": 1447, "lon": 2895}, {"phony_dim_0": 300}, {"time": None, "lat": None, "lon": None}
        """
        size_attr = self._get_attrib(root, "size")
        name_attr = self._get_attrib(root, "name")

        if size_attr is not None:
            size = int(size_attr)
            if name_attr is not None:
                return {Path(name_attr).name: size}
            else:
                return {f"phony_dim_{dim_index}": size}

        raise ValueError("Not enough information to parse Dim/Dimension tag")

    def _find_dimension_tags(self, root: ET.Element) -> list[ET.Element]:
        """
        Find the all tags with dimension information.

        First attempts to find Dimension tags, then falls back to Dim tags.
        If Dim tags are found, the fully qualified name is used to find the corresponding Dimension tag.
        If Dim tags have no name attribute, they are phony dimensions and used directly.

        Parameters
        ----------
        root : ET.Element
            An ElementTree Element from a DMR++ file.

        Returns
        -------
        list[ET.Element]
        """
        dimension_tags = root.findall("dap:Dimension", self._NS)
        if not dimension_tags:
            # Dim tags contain a fully qualified name that references a Dimension tag elsewhere in the DMR++
            # or they are phony dimensions (have size but no name)
            dim_tags = root.findall("dap:Dim", self._NS)
            for d in dim_tags:
                dim_name = self._get_attrib(d, "name")
                if dim_name is not None:
                    dimension_tag = self.find_node_fqn(dim_name)
                    if dimension_tag is not None:
                        dimension_tags.append(dimension_tag)
                else:
                    # Phony dimension - use the Dim tag directly
                    dimension_tags.append(d)
        return dimension_tags

    def _parse_variable(self, var_tag: ET.Element) -> ManifestArray:
        """
        Parse a variable from a DMR++ tag.

        Parameters
        ----------
        var_tag : ET.Element
            An ElementTree Element representing a variable in the DMR++ file. Will have DAP dtype as tag. E.g. <Float32>

        Returns
        -------
        ManifestArray
        """

        # Dimension info
        dims: dict[str, int] = {}
        dimension_tags = self._find_dimension_tags(var_tag)
        for dim_index, dim in enumerate(dimension_tags):
            dims.update(self._parse_dim(dim, dim_index=dim_index))
        # convert DAP dtype to numpy dtype
        dtype = np.dtype(
            self._DAP_NP_DTYPE[var_tag.tag.removeprefix("{" + self._NS["dap"] + "}")]
        )
        # Chunks and Filters
        shape: tuple[int, ...] = tuple(dims.values())
        chunks_shape = shape
        chunks_tag = var_tag.find("dmrpp:chunks", self._NS)
        array_fill_value = np.array(0).astype(dtype)[()]
        if chunks_tag is not None:
            # Chunks
            chunk_dim_text = chunks_tag.findtext(
                "dmrpp:chunkDimensionSizes", namespaces=self._NS
            )
            if chunk_dim_text is not None:
                # 1 1447 2895 -> (1, 1447, 2895)
                chunks_shape = tuple(map(int, chunk_dim_text.split()))
            else:
                chunks_shape = shape
            fill_value = self._get_attrib(chunks_tag, "fillValue")
            if fill_value is not None:
                array_fill_value = np.array(fill_value).astype(dtype)[()]
            if chunks_shape:
                chunkmanifest = self._parse_chunks(chunks_tag, chunks_shape)
            else:
                chunkmanifest = ChunkManifest(entries={}, shape=array_fill_value.shape)
            # Filters
            codecs = self._parse_filters(chunks_tag, dtype)

        # Attributes
        attrs: dict[str, Any] = {}
        for attr_tag in var_tag.iterfind("dap:Attribute", self._NS):
            attrs.update(self._parse_attribute(attr_tag))
        if "_FillValue" in attrs:
            encoded_cf_fill_value = encode_cf_fill_value(attrs["_FillValue"], dtype)
            attrs["_FillValue"] = encoded_cf_fill_value

        metadata = create_v3_array_metadata(
            shape=shape,
            data_type=dtype,
            chunk_shape=chunks_shape,
            codecs=codecs,
            dimension_names=dims,
            attributes=attrs,
            fill_value=array_fill_value,
        )
        return ManifestArray(metadata=metadata, chunkmanifest=chunkmanifest)

    def _parse_attribute(self, attr_tag: ET.Element) -> dict[str, Any]:
        """
        Parse an attribute from a DMR++ attr tag. Converts the attribute value to a native python type.
        Raises an exception if nested attributes are passed. Container attributes must be unwrapped in the parent function.

        Parameters
        ----------
        attr_tag : ET.Element
            An ElementTree Element with an <Attr> tag.

        Returns
        -------
        dict
        """
        attr: dict[str, Any] = {}
        values = []
        attr_type = self._get_attrib(attr_tag, "type")
        attr_name = self._get_attrib(attr_tag, "name", required=True)

        if attr_name is None:
            return {}

        if attr_type == "Container":
            # DMR++ build information that is not part of the dataset
            if attr_name == "build_dmrpp_metadata":
                return {}
            else:
                warnings.warn(
                    "This DMRpp contains a nested attribute "
                    f"{attr_name}. Nested attributes cannot "
                    "be assigned to a variable or dataset and will be dropped"
                )
                return {}

        if attr_type is None:
            return {}

        dtype = np.dtype(self._DAP_NP_DTYPE[attr_type])
        # if multiple Value tags are present, store as "key": "[v1, v2, ...]"
        for value_tag in attr_tag:
            # cast attribute to native python type using dmr provided dtype
            val = (
                dtype.type(value_tag.text).item()
                if dtype != np.object_
                else value_tag.text
            )
            # "*" may represent nan values in DMR++
            if val == "*":
                val = np.nan
            values.append(val)
        attr[attr_name] = values[0] if len(values) == 1 else values
        return attr

    def _parse_filters(
        self, chunks_tag: ET.Element, dtype: np.dtype
    ) -> list[dict] | None:
        """
        Parse filters from a DMR++ chunks tag.

        Parameters
        ----------
        chunks_tag : ET.Element
            An ElementTree Element with a <chunks> tag.

        dtype : np.dtype
            The numpy dtype of the variable.

        Returns
        -------
        list[dict] | None
            E.g. [{"id": "shuffle", "elementsize": 4}, {"id": "zlib", "level": 4}]
        """
        compression_type = self._get_attrib(chunks_tag, "compressionType")
        if compression_type is not None:
            filters: list[dict] = []
            # shuffle deflate --> ["shuffle", "deflate"]
            compression_types = compression_type.split(" ")
            for c in compression_types:
                if c == "shuffle":
                    filters.append(
                        {
                            "name": "numcodecs.shuffle",
                            "configuration": {"elementsize": dtype.itemsize},
                        }
                    )
                elif c == "deflate":
                    deflate_level = self._get_attrib(chunks_tag, "deflateLevel")
                    level = (
                        int(deflate_level)
                        if deflate_level is not None
                        else self._DEFAULT_ZLIB_VALUE
                    )
                    filters.append(
                        {
                            "name": "numcodecs.zlib",
                            "configuration": {
                                "level": level,
                            },
                        }
                    )
            return filters
        return None

    def _parse_chunks(
        self, chunks_tag: ET.Element, chunks_shape: tuple[int, ...]
    ) -> ChunkManifest:
        """
        Parse the chunk manifest from a DMR++ chunks tag.

        Parameters
        ----------
        chunks_tag : ET.Element
            An ElementTree Element with a <chunks> tag.

        chunks_shape : tuple
            Chunk sizes for each dimension. E.g. (1, 1447, 2895)

        Returns
        -------
        ChunkManifest
        """
        chunkmanifest: dict[ChunkKey, object] = {}
        default_num: list[int] = (
            [0 for i in range(len(chunks_shape))] if chunks_shape else [0]
        )
        chunk_key_template = ".".join(["{}" for i in range(len(default_num))])
        for chunk_tag in chunks_tag.iterfind("dmrpp:chunk", self._NS):
            chunk_num = default_num
            chunk_position = self._get_attrib(chunk_tag, "chunkPositionInArray")
            if chunk_position is not None:
                # "[0,1023,10235]" -> ["0","1023","10235"]
                chunk_pos = chunk_position[1:-1].split(",")
                # [0,1023,10235] // [1, 1023, 2047] -> [0,1,5]
                chunk_num = [
                    int(chunk_pos[i]) // chunks_shape[i]
                    for i in range(len(chunks_shape))
                ]
            # [0,1,5] -> "0.1.5"
            chunk_key = ChunkKey(chunk_key_template.format(*chunk_num))
            offset = self._get_attrib(chunk_tag, "offset", required=True)
            n_bytes = self._get_attrib(chunk_tag, "nBytes", required=True)
            if offset is not None and n_bytes is not None:
                chunkmanifest[chunk_key] = {
                    "path": self.data_filepath,
                    "offset": int(offset),
                    "length": int(n_bytes),
                }
        return ChunkManifest(entries=chunkmanifest)
