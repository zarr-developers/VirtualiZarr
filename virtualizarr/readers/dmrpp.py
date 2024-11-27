import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

import numpy as np
from xarray import Coordinates, Dataset, Index, Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.readers.common import VirtualBackend
from virtualizarr.types import ChunkKey
from virtualizarr.utils import _FsspecFSFromFilepath, check_for_collisions
from virtualizarr.zarr import ZArray


class DMRPPVirtualBackend(VirtualBackend):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        loadable_variables, drop_variables = check_for_collisions(
            drop_variables=drop_variables,
            loadable_variables=loadable_variables,
        )

        if virtual_backend_kwargs:
            raise NotImplementedError(
                "DMR++ reader does not understand any virtual_backend_kwargs"
            )

        if loadable_variables != [] or decode_times or indexes is None:
            raise NotImplementedError(
                "Specifying `loadable_variables` or auto-creating indexes with `indexes=None` is not supported for dmrpp files."
            )

        filepath = validate_and_normalize_path_to_uri(
            filepath, fs_root=Path.cwd().as_uri()
        )

        fpath = _FsspecFSFromFilepath(
            filepath=filepath, reader_options=reader_options
        ).open_file()

        parser = DMRParser(
            root=ET.parse(fpath).getroot(),
            data_filepath=filepath.removesuffix(".dmrpp"),
        )
        vds = parser.parse_dataset(group=group, indexes=indexes)

        return vds.drop_vars(drop_variables)


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
    _ENCODING_KEYS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}
    root: ET.Element
    data_filepath: str

    def __init__(self, root: ET.Element, data_filepath: Optional[str] = None):
        """
        Initialize the DMRParser with the given DMR++ file contents and source data file path.

        Parameters
        ----------
        root: xml.ElementTree.Element
            Root of the xml tree struture of a DMR++ file.
        data_filepath : str, optional
            The path to the actual data file that will be set in the chunk manifests.
            If None, the data file path is taken from the DMR++ file.
        """
        self.root = root
        self.data_filepath = (
            data_filepath if data_filepath is not None else self.root.attrib["name"]
        )

    def parse_dataset(self, group=None, indexes: Mapping[str, Index] = {}) -> Dataset:
        """
        Parses the given file and creates a virtual xr.Dataset with ManifestArrays.

        Parameters
        ----------
        group : str
            The group to parse. If None, and no groups are present, the dataset is parsed.
            If None and groups are present, the first group is parsed.

        indexes : Mapping[str, Index], default is {}
            Indexes to use on the returned xarray Dataset.
            Default is {} which will avoid creating any indexes

        Returns
        -------
        An xr.Dataset wrapping virtualized zarr arrays.

        Examples
        --------
        Open a sample DMR++ file and parse the dataset

        >>> import requests
        >>> r = requests.get("https://github.com/OPENDAP/bes/raw/3e518f6dc2f625b0b83cfb6e6fd5275e4d6dcef1/modules/dmrpp_module/data/dmrpp/chunked_threeD.h5.dmrpp")
        >>> parser = DMRParser(r.text)
        >>> vds = parser.parse_dataset()
        >>> vds
        <xarray.Dataset> Size: 4MB
            Dimensions:     (phony_dim_0: 100, phony_dim_1: 100, phony_dim_2: 100)
            Dimensions without coordinates: phony_dim_0, phony_dim_1, phony_dim_2
            Data variables:
                d_8_chunks  (phony_dim_0, phony_dim_1, phony_dim_2) float32 4MB ManifestA...

        >>> vds2 = open_virtual_dataset("https://github.com/OPENDAP/bes/raw/3e518f6dc2f625b0b83cfb6e6fd5275e4d6dcef1/modules/dmrpp_module/data/dmrpp/chunked_threeD.h5.dmrpp", filetype="dmrpp", indexes={})
        >>> vds2
        <xarray.Dataset> Size: 4MB
            Dimensions:     (phony_dim_0: 100, phony_dim_1: 100, phony_dim_2: 100)
            Dimensions without coordinates: phony_dim_0, phony_dim_1, phony_dim_2
            Data variables:
                d_8_chunks  (phony_dim_0, phony_dim_1, phony_dim_2) float32 4MB ManifestA...
        """
        group_tags = self.root.findall("dap:Group", self._NS)
        if group is not None:
            group = Path(group)
            if not group.is_absolute():
                group = Path("/") / group
            if len(group_tags) == 0:
                warnings.warn("No groups found in DMR++ file; ignoring group parameter")
            else:
                all_groups = self._split_groups(self.root)
                if group in all_groups:
                    return self._parse_dataset(all_groups[group], indexes)
                else:
                    raise ValueError(f"Group {group} not found in DMR++ file")
        return self._parse_dataset(self.root, indexes)

    def find_node_fqn(self, fqn: str) -> ET.Element:
        """
        Find the element in the root element by converting the fully qualified name to an xpath query.

        E.g. fqn = "/a/b" --> root.find("./*[@name='a']/*[@name='b']")
        See more about OPeNDAP fully qualified names (FQN) here: https://docs.opendap.org/index.php/DAP4:_Specification_Volume_1#Fully_Qualified_Names

        Parameters
        ----------
        fqn : str
            The fully qualified name of an element. E.g. "/a/b"

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
        xpath_query = "./" + "/".join(xpath_segments)  # "./[*[@name='a']/*[@name='b']"
        element = self.root.find(xpath_query, self._NS)
        if element is None:
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
            new_path = current_path / Path(g.attrib["name"])
            dataset_tags = [
                d for d in g if d.tag != "{" + self._NS["dap"] + "}" + "Group"
            ]
            group_dict[new_path] = ET.Element(g.tag, g.attrib)
            group_dict[new_path].extend(dataset_tags)
            group_dict.update(self._split_groups_recursive(g, new_path))
        return group_dict

    def _parse_dataset(
        self, root: ET.Element, indexes: Mapping[str, Index] = {}
    ) -> Dataset:
        """
        Parse the dataset using the root element of the DMR++ file.

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR++ file.

        Returns
        -------
        xr.Dataset
        """
        # Dimension names and sizes
        dims: dict[str, int] = {}
        dimension_tags = self._find_dimension_tags(root)
        for dim in dimension_tags:
            dims.update(self._parse_dim(dim))
        # Data variables and coordinates
        coord_names: set[str] = set()
        coord_tags = root.findall(
            ".//dap:Attribute[@name='coordinates']/dap:Value", self._NS
        )
        for c in coord_tags:
            if c.text is not None:
                coord_names.update(c.text.split(" "))
        # Seperate and parse coords + data variables
        coord_vars: dict[str, Variable] = {}
        data_vars: dict[str, Variable] = {}
        for var_tag in self._find_var_tags(root):
            variable = self._parse_variable(var_tag)
            # Either coordinates are explicitly defined or 1d variable with same name as dimension is a coordinate
            if var_tag.attrib["name"] in coord_names or (
                len(variable.dims) == 1 and variable.dims[0] == var_tag.attrib["name"]
            ):
                coord_vars[var_tag.attrib["name"]] = variable
            else:
                data_vars[var_tag.attrib["name"]] = variable
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
        return Dataset(
            data_vars=data_vars,
            coords=Coordinates(coords=coord_vars, indexes=indexes),
            attrs=attrs,
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

    def _parse_dim(self, root: ET.Element) -> dict[str, int]:
        """
        Parse single <Dim> or <Dimension> tag

        If the tag has no name attribute, it is a phony dimension. E.g. <Dim size="300"/> --> {"phony_dim": 300}
        If the tag has both name and size attributes, it is a regular dimension. E.g. <Dim name="lat" size="1447"/> --> {"lat": 1447}

        Parameters
        ----------
        root : ET.Element
            The root element Dim/Dimension tag

        Returns
        -------
        dict
            E.g. {"time": 1, "lat": 1447, "lon": 2895}, {"phony_dim": 300}, {"time": None, "lat": None, "lon": None}
        """
        if "name" not in root.attrib and "size" in root.attrib:
            return {"phony_dim": int(root.attrib["size"])}
        if "name" in root.attrib and "size" in root.attrib:
            return {Path(root.attrib["name"]).name: int(root.attrib["size"])}
        raise ValueError("Not enough information to parse Dim/Dimension tag")

    def _find_dimension_tags(self, root: ET.Element) -> list[ET.Element]:
        """
        Find the all tags with dimension information.

        First attempts to find Dimension tags, then falls back to Dim tags.
        If Dim tags are found, the fully qualified name is used to find the corresponding Dimension tag.

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
            dim_tags = root.findall("dap:Dim", self._NS)
            for d in dim_tags:
                dimension_tag = self.find_node_fqn(d.attrib["name"])
                if dimension_tag is not None:
                    dimension_tags.append(dimension_tag)
        return dimension_tags

    def _parse_variable(self, var_tag: ET.Element) -> Variable:
        """
        Parse a variable from a DMR++ tag.

        Parameters
        ----------
        var_tag : ET.Element
            An ElementTree Element representing a variable in the DMR++ file. Will have DAP dtype as tag. E.g. <Float32>

        Returns
        -------
        xr.Variable
        """
        # Dimension info
        dims: dict[str, int] = {}
        dimension_tags = self._find_dimension_tags(var_tag)
        if not dimension_tags:
            raise ValueError("Variable has no dimensions")
        for dim in dimension_tags:
            dims.update(self._parse_dim(dim))
        # convert DAP dtype to numpy dtype
        dtype = np.dtype(
            self._DAP_NP_DTYPE[var_tag.tag.removeprefix("{" + self._NS["dap"] + "}")]
        )
        # Chunks and Filters
        filters = None
        shape: tuple[int, ...] = tuple(dims.values())
        chunks_shape = shape
        chunks_tag = var_tag.find("dmrpp:chunks", self._NS)
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
            chunkmanifest = self._parse_chunks(chunks_tag, chunks_shape)
            # Filters
            filters = self._parse_filters(chunks_tag, dtype)
        # Attributes
        attrs: dict[str, Any] = {}
        for attr_tag in var_tag.iterfind("dap:Attribute", self._NS):
            attrs.update(self._parse_attribute(attr_tag))
        # Fill value is placed in encoding and thus removed from attributes
        fill_value = attrs.pop("_FillValue", None)
        # create ManifestArray and ZArray
        zarray = ZArray(
            chunks=chunks_shape,
            dtype=dtype,
            fill_value=fill_value,
            filters=filters,
            order="C",
            shape=shape,
        )
        marr = ManifestArray(zarray=zarray, chunkmanifest=chunkmanifest)
        encoding = {k: attrs.get(k) for k in self._ENCODING_KEYS if k in attrs}
        return Variable(dims=dims.keys(), data=marr, attrs=attrs, encoding=encoding)

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
        if "type" in attr_tag.attrib and attr_tag.attrib["type"] == "Container":
            # DMR++ build information that is not part of the dataset
            if attr_tag.attrib["name"] == "build_dmrpp_metadata":
                return {}
            raise ValueError(
                "Nested attributes cannot be assigned to a variable or dataset"
            )
        dtype = np.dtype(self._DAP_NP_DTYPE[attr_tag.attrib["type"]])
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
        attr[attr_tag.attrib["name"]] = values[0] if len(values) == 1 else values
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
        if "compressionType" in chunks_tag.attrib:
            filters: list[dict] = []
            # shuffle deflate --> ["shuffle", "deflate"]
            compression_types = chunks_tag.attrib["compressionType"].split(" ")
            for c in compression_types:
                if c == "shuffle":
                    filters.append({"id": "shuffle", "elementsize": dtype.itemsize})
                elif c == "deflate":
                    filters.append(
                        {
                            "id": "zlib",
                            "level": int(
                                chunks_tag.attrib.get(
                                    "deflateLevel", self._DEFAULT_ZLIB_VALUE
                                )
                            ),
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
            if "chunkPositionInArray" in chunk_tag.attrib:
                # "[0,1023,10235]" -> ["0","1023","10235"]
                chunk_pos = chunk_tag.attrib["chunkPositionInArray"][1:-1].split(",")
                # [0,1023,10235] // [1, 1023, 2047] -> [0,1,5]
                chunk_num = [
                    int(chunk_pos[i]) // chunks_shape[i]
                    for i in range(len(chunks_shape))
                ]
            # [0,1,5] -> "0.1.5"
            chunk_key = ChunkKey(chunk_key_template.format(*chunk_num))
            chunkmanifest[chunk_key] = {
                "path": self.data_filepath,
                "offset": int(chunk_tag.attrib["offset"]),
                "length": int(chunk_tag.attrib["nBytes"]),
            }
        return ChunkManifest(entries=chunkmanifest)
