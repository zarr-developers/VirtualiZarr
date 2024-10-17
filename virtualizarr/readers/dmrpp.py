import os
import warnings
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import numpy as np
import xarray as xr
from xarray.core.indexes import Index

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.types import ChunkKey
from virtualizarr.zarr import ZArray


class DMRParser:
    """
    Parser for the OPeNDAP DMR++ XML format.
    Reads groups, dimensions, coordinates, data variables, encoding, chunk manifests, and attributes.
    Highly modular to allow support for older dmrpp schema versions. Includes many utility functions to extract
    different information such as finding all variable tags, splitting hdf5 groups, parsing dimensions, and more.

    OPeNDAP DMR++ homepage: https://docs.opendap.org/index.php/DMR%2B%2B
    """

    # DAP and DMRPP XML namespaces
    _ns = {
        "dap": "http://xml.opendap.org/ns/DAP/4.0#",
        "dmrpp": "http://xml.opendap.org/dap/dmrpp/1.0.0#",
    }
    # DAP data types to numpy data types
    _dap_np_dtype = {
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
    _default_zlib_value = 6
    # Encoding keys that should be removed from attributes and placed in xarray encoding dict
    _encoding_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

    def __init__(self, dmrpp_str: str, data_filepath: Optional[str] = None):
        """
        Initialize the DMRParser with the given DMR++ file contents and source data file path.

        Parameters
        ----------
        dmrpp_str : str
            The dmrpp file contents as a string.

        data_filepath : str, optional
            The path to the actual data file that will be set in the chunk manifests.
            If None, the data file path is taken from the DMR++ file.
        """
        self.root = ET.fromstring(dmrpp_str)
        self.data_filepath = (
            data_filepath if data_filepath is not None else self.root.attrib["name"]
        )

    def parse_dataset(
        self, group=None, indexes: Mapping[str, Index] = {}
    ) -> xr.Dataset:
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
        if group is not None:
            group = Path(group)
        group_tags = self.root.findall("dap:Group", self._ns)
        if len(group_tags) == 0:
            if group is not None:
                # no groups found and group specified -> warning
                warnings.warn("No groups found in DMR++ file; ignoring group parameter")
            # no groups found -> parse dataset
            return self._parse_dataset(self.root, indexes)
        all_groups = self._split_groups(self.root)
        if group is None:
            # groups found and no group specified -> parse root group
            return self._parse_dataset(self.root, indexes)
        if group.name in all_groups:
            # groups found and group specified -> parse specified group
            return self._parse_dataset(all_groups[group.name], indexes)
        else:
            # groups found and specified group not found -> error
            raise ValueError(f"Group {group.name} not found in DMR++ file")

    def _fqn_xpath(self, fqn: str) -> str:
        """
        Create a fully qualified xpath from the root element and a fully qualified name.

        Parameters
        ----------
        fqn : str
            The fully qualified name to create an xpath for.

        Returns
        -------
        str
        """
        if fqn == "":
            return "."
        elements = fqn.strip("/").split("/")  # /a/b/ --> ['a', 'b']
        xpath_segments = [f"*[@name='{element}']" for element in elements]
        return "./" + "/".join(xpath_segments)  # "./[*[@name='a']/*[@name='b']"

    def _split_groups(self, root: ET.Element) -> dict[str, ET.Element]:
        """
        Split the input <Dataset> element into several <Dataset> ET.Elements by <Group> name.
        E.g. {"/": <Dataset>, "left": <Dataset>, "right": <Dataset>}

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR file.

        Returns
        -------
        dict[str, ET.Element]
        """
        all_groups: dict[str, ET.Element] = defaultdict(
            lambda: ET.Element(root.tag, root.attrib)
        )
        dataset_tags = [
            d for d in root if d.tag != "{" + self._ns["dap"] + "}" + "Group"
        ]
        if len(dataset_tags) > 0:
            all_groups["/"].extend(dataset_tags)
        all_groups.update(self._split_groups_recursive(root))
        return all_groups

    def _split_groups_recursive(
        self, root: ET.Element, current_path=Path("")
    ) -> dict[Path, ET.Element]:
        group_dict = defaultdict(lambda: ET.Element(root.tag, root.attrib))
        for g in root.iterfind("dap:Group", self._ns):
            new_path = str(current_path / g.attrib["name"])
            dataset_tags = [
                d for d in g if d.tag != "{" + self._ns["dap"] + "}" + "Group"
            ]
            group_dict[new_path].extend(dataset_tags)
            group_dict.update(self._split_groups_recursive(g, new_path))
        return group_dict

    def _parse_dataset(
        self, root: ET.Element, indexes: Mapping[str, Index] = {}
    ) -> xr.Dataset:
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
        # if not dimension_tags:
        # raise ValueError("Dataset has no dimensions")
        for dim in dimension_tags:
            dims.update(self._parse_dim(dim))
        # Data variables and coordinates
        coord_names = self._find_coord_names(root)
        # if no coord_names are found or coords don't include dims, dims are used as coords
        if len(coord_names) == 0 or len(coord_names) < len(dims):
            coord_names = set(dims.keys())
        # Seperate and parse coords + data variables
        coord_vars: dict[str, xr.Variable] = {}
        data_vars: dict[str, xr.Variable] = {}
        for var_tag in self._find_var_tags(root):
            variable = self._parse_variable(var_tag)
            if var_tag.attrib["name"] in coord_names:
                coord_vars[var_tag.attrib["name"]] = variable
            else:
                data_vars[var_tag.attrib["name"]] = variable
        # Attributes
        attrs: dict[str, str] = {}
        for attr_tag in root.iterfind("dap:Attribute", self._ns):
            attrs.update(self._parse_attribute(attr_tag))
        return xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates(coords=coord_vars, indexes=indexes),
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
        for dap_dtype in self._dap_np_dtype:
            vars_tags += root.findall(f"dap:{dap_dtype}", self._ns)
        return vars_tags

    def _find_coord_names(self, root: ET.Element) -> set[str]:
        """
        Find the name of all coordinates in root. Checks inside all variables and global attributes.

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR++ file.

        Returns
        -------
        set[str] : The set of unique coordinate names.
        """
        # Check for coordinate names within each variable attributes
        coord_names: set[str] = set()
        for var_tag in self._find_var_tags(root):
            coord_text = var_tag.findtext(
                "./dap:Attribute[@name='coordinates']/dap:Value", namespaces=self._ns
            )
            if coord_text is not None:
                coord_names.update(coord_text.split(" "))
            for map_tag in var_tag.iterfind("dap:Map", self._ns):
                coord_names.add(Path(map_tag.attrib["name"]).name)
        # Check for coordinate names in a global attribute
        global_coord_text = root.findtext(
            "./dap:Attribute[@name='coordinates']", namespaces=self._ns
        )
        if global_coord_text is not None:
            coord_names.update(global_coord_text.split(" "))
        return coord_names

    def _parse_dim(self, root: ET.Element) -> dict[str, int | None]:
        """
        Parse single <Dim> or <Dimension> tag

        If the tag has no name attribute, it is a phony dimension. E.g. <Dim size="300"/> --> {"phony_dim": 300}
        If the tag has no size attribute, it is an unlimited dimension. E.g. <Dim name="time"/> --> {"time": None}
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
        if "name" in root.attrib and "size" not in root.attrib:
            return {os.path.basename(root.attrib["name"]): None}
        if "name" in root.attrib and "size" in root.attrib:
            return {os.path.basename(root.attrib["name"]): int(root.attrib["size"])}
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
        dimension_tags = root.findall("dap:Dimension", self._ns)
        if not dimension_tags:
            # Dim tags contain a fully qualified name that references a Dimension tag elsewhere in the DMR++
            dim_tags = root.findall("dap:Dim", self._ns)
            for d in dim_tags:
                dimension_tag = self.root.find(self._fqn_xpath(d.attrib["name"]))
                if dimension_tag is not None:
                    dimension_tags.append(dimension_tag)
        return dimension_tags

    def _parse_variable(self, var_tag: ET.Element) -> xr.Variable:
        """
        Parse a variable from a DMR++ tag.

        Parameters
        ----------
        var_tag : ET.Element
            An ElementTree Element representing a variable in the DMR++ file. Will have DAP dtype as tag. E.g. <Float32>

        dataset_dims : dict
            A dictionary of dimension names and sizes. E.g. {"time": 1, "lat": 1447, "lon": 2895}
            Must contain at least all the dimensions used by the variable. Necessary since the variable
            metadata only contains the dimension names and not the sizes.

        Returns
        -------
        xr.Variable
        """
        # Dimension info
        dims: {str, int} = {}
        dimension_tags = self._find_dimension_tags(var_tag)
        if not dimension_tags:
            raise ValueError("Variable has no dimensions")
        for dim in dimension_tags:
            dims.update(self._parse_dim(dim))
        # convert DAP dtype to numpy dtype
        dtype = np.dtype(
            self._dap_np_dtype[var_tag.tag.removeprefix("{" + self._ns["dap"] + "}")]
        )
        # Chunks and Filters
        filters = None
        shape: tuple[int, ...] = tuple(dims.values())
        chunks_shape = shape
        chunks_tag = var_tag.find("dmrpp:chunks", self._ns)
        if chunks_tag is not None:
            # Chunks
            chunk_dim_text = chunks_tag.findtext(
                "dmrpp:chunkDimensionSizes", namespaces=self._ns
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
        for attr_tag in var_tag.iterfind("dap:Attribute", self._ns):
            attrs.update(self._parse_attribute(attr_tag))
        # Fill value is placed in encoding and thus removed from attributes
        fill_value = attrs.pop("_FillValue", np.nan)
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
        encoding = {k: attrs.get(k) for k in self._encoding_keys if k in attrs}
        return xr.Variable(dims=dims.keys(), data=marr, attrs=attrs, encoding=encoding)

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
        dtype = np.dtype(self._dap_np_dtype[attr_tag.attrib["type"]])
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
                                    "deflateLevel", self._default_zlib_value
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
        for chunk_tag in chunks_tag.iterfind("dmrpp:chunk", self._ns):
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
