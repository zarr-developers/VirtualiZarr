from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.manifest import validate_chunk_keys
from virtualizarr.types import ChunkKey
from virtualizarr.zarr import ZArray


class DMRParser:
    """
    Parses a DMR file and creates a virtual xr.Dataset.
    Handles groups, dimensions, coordinates, data variables, encoding, chunk manifests, and attributes.
    """

    _ns = {
        "dap": "http://xml.opendap.org/ns/DAP/4.0#",
        "dmr": "http://xml.opendap.org/dap/dmrpp/1.0.0#",
    }
    dap_np_dtype = {
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

    def __init__(self, dmr: str, data_filepath: Optional[str] = None):
        """
        Initialize the DMRParser with the given DMR data and data file path.

        Parameters
        ----------
        dmr : str
            The DMR file contents as a string.

        data_filepath : str, optional
            The path to the actual data file that will be set in the chunk manifests.
            If None, the data file path is taken from the DMR file.
        """
        self.root = ET.fromstring(dmr)
        self.data_filepath = (
            data_filepath if data_filepath is not None else self.root.attrib["name"]
        )
        self._global_dims: dict[str, int] = {}
        self._group: str | None = None

    def parse(self, group: Optional[str] = None) -> xr.Dataset:
        """
        Parse the dataset from the dmrpp file

        Parameters
        ----------
        group : str
            The group to parse. If None, the entire dataset is parsed.

        Returns
        -------
        An xr.Dataset wrapping virtualized zarr arrays.
        """
        self._group = group
        if self._group is not None:
            self._group = (
                "/" + self._group.strip("/") + "/"
            )  # ensure group is in form "/a/b/"
        if self.data_filepath.endswith(".h5"):
            return self._parse_hdf5_dataset()
        group_tags = self.root.findall("dap:Group", self._ns)
        if len(group_tags) > 0 and self._group is not None:
            return self._parse_netcdf4_group(group_tags)
        return self._parse_dataset()

    def _parse_netcdf4_group(self, group_tags: list[ET.Element]) -> xr.Dataset:
        """
        Parse the dataset from the netcdf4 based dmrpp with groups, starting at the given group.
        Set root to the given group.

        Parameters
        ----------
        group_tags : list[ET.element]
            A list of ET elements representing the groups in the DMR file.
            Each will be a <Group> tag.
        Returns
        -------
        xr.Dataset
        """
        self.root = group_tags[0]
        for group_tag in group_tags:
            if self._group is not None and group_tag.attrib[
                "name"
            ] == self._group.strip("/"):
                self.root = group_tag
        return self._parse_dataset()

    def _parse_hdf5_dataset(self) -> xr.Dataset:
        """
        Parse the dataset from the HDF5 based dmrpp, starting at the given group.
        Set root to the given group.

        Returns
        -------
        xr.Dataset
        """
        if self._group is None:
            # NOTE: This will return an xr.DataTree with all groups in the future...
            raise ValueError("HDF5 based DMR parsing requires a group to be specified")
        # Make a new root containing only dims, vars, and attrs for the dataset specified by group
        ds_root = ET.Element(self.root.tag, self.root.attrib)
        dim_names: set[str] = set()
        vars_tags: list[ET.Element] = []
        orignames = {}  # store original names for renaming later
        for dap_dtype in self.dap_np_dtype:
            vars_tags += self.root.findall(f"dap:{dap_dtype}", self._ns)
        # Add variables part of group to ds_root
        for var_tag in vars_tags:
            fullname_tag = var_tag.find(
                "./dap:Attribute[@name='fullnamepath']/dap:Value", self._ns
            )
            origname_tag = var_tag.find(
                "./dap:Attribute[@name='origname']/dap:Value", self._ns
            )
            if (
                fullname_tag is not None
                and origname_tag is not None
                and fullname_tag.text is not None
                and origname_tag.text is not None
                and fullname_tag.text == self._group + origname_tag.text
            ):
                ds_root.append(var_tag)
                orignames[var_tag.attrib["name"]] = origname_tag.text
                for dim_tag in var_tag.findall("dap:Dim", self._ns):
                    dim_names.add(dim_tag.attrib["name"][1:])
        # Add dimensions part of group to root2
        for dim_tag in self.root.iterfind("dap:Dimension", self._ns):
            if dim_tag.attrib["name"] in dim_names:
                ds_root.append(dim_tag)
        # make an empty xml element
        container_attr_tag: ET.Element = ET.Element("Attribute")
        for attr_tag in self.root.findall("dap:Attribute", self._ns):
            fullname_tag = attr_tag.find(
                "./dap:Attribute[@name='fullnamepath']/dap:Value", self._ns
            )
            if fullname_tag is not None and fullname_tag.text == self._group[:-1]:
                container_attr_tag = attr_tag
        # add all attributes for the group to the new root (except fullnamepath)
        ds_root.extend(
            [a for a in container_attr_tag if a.attrib["name"] != "fullnamepath"]
        )
        self.root = ds_root
        return self._parse_dataset().rename(orignames)

    def _parse_dataset(self) -> xr.Dataset:
        """
        Parse the dataset using the root element of the DMR file.

        Returns
        -------
        xr.Dataset
        """
        # find all dimension names and sizes
        for dim_tag in self.root.iterfind("dap:Dimension", self._ns):
            self._global_dims[dim_tag.attrib["name"]] = int(dim_tag.attrib["size"])
        vars_tags: list[ET.Element] = []
        for dap_dtype in self.dap_np_dtype:
            vars_tags += self.root.findall(f"dap:{dap_dtype}", self._ns)
        # find all coordinate names (using Map tags and coordinates attribute)
        coord_names: set[str] = set()
        for var_tag in vars_tags:
            coord_tag = var_tag.find(
                "./dap:Attribute[@name='coordinates']/dap:Value", self._ns
            )
            if coord_tag is not None and coord_tag.text is not None:
                coord_names.update(coord_tag.text.split(" "))
            for map_tag in var_tag.iterfind("dap:Map", self._ns):
                coord_names.add(map_tag.attrib["name"].removeprefix("/"))
        # if no coord_names are found or coords don't include dims, dims are used as coords
        if len(coord_names) == 0 or len(coord_names) < len(self._global_dims):
            coord_names = set(self._global_dims.keys())
        # find all coords + data variables
        coords: dict[str, xr.Variable] = {}
        data_vars: dict[str, xr.Variable] = {}
        for var_tag in vars_tags:
            if var_tag.attrib["name"] in coord_names:
                coords[var_tag.attrib["name"]] = self._parse_variable(var_tag)
            else:
                data_vars[var_tag.attrib["name"]] = self._parse_variable(var_tag)
        # find all dataset attributes
        attrs: dict[str, str] = {}
        for attr_tag in self.root.iterfind("dap:Attribute", self._ns):
            if attr_tag.attrib["type"] != "Container":  # container = nested attributes
                attrs.update(self._parse_attribute(attr_tag))
        return xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates(coords=coords, indexes={}),
            attrs=attrs,
        )

    def _parse_variable(self, var_tag: ET.Element) -> xr.Variable:
        """
        Parse a variable from a DMR tag.

        Parameters
        ----------
        var_tag : ET.Element
            An ElementTree Element representing a variable in the DMR file. Will have DAP dtype as tag.

        Returns
        -------
        xr.Variable
        """
        # parse dimensions
        dims: list[str] = []
        for dim_tag in var_tag.iterfind("dap:Dim", self._ns):
            dim = (
                dim_tag.attrib["name"]
                if self._group is None
                else dim_tag.attrib["name"].removeprefix(self._group)
            )
            dims.append(dim.removeprefix("/"))
        shape = tuple([self._global_dims[d] for d in dims])
        # parse chunks
        chunks = shape
        chunks_tag = var_tag.find("dmr:chunks", self._ns)
        if chunks_tag is None:
            raise ValueError(
                f"No chunks tag found in DMR file for variable {var_tag.attrib['name']}"
            )
        chunk_dim_tag = chunks_tag.find("dmr:chunkDimensionSizes", self._ns)
        if chunk_dim_tag is not None and chunk_dim_tag.text is not None:
            chunks = tuple(
                map(int, chunk_dim_tag.text.split())
            )  # 1 1447 2895 -> (1, 1447, 2895)
        chunkmanifest = self._parse_chunks(chunks_tag, chunks)
        # parse attributes
        attrs: dict[str, str] = {}
        for attr_tag in var_tag.iterfind("dap:Attribute", self._ns):
            attrs.update(self._parse_attribute(attr_tag))
        # create ManifestArray and ZArray
        # convert DAP dtype to numpy dtype
        dtype = np.dtype(
            self.dap_np_dtype[var_tag.tag.removeprefix("{" + self._ns["dap"] + "}")]
        )
        fill_value = (
            attrs["_FillValue"]
            if "_FillValue" in attrs and attrs["_FillValue"] != "*"
            else None
        )
        zarray = ZArray(
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            order="C",
            shape=shape,
            zarr_format=3,
        )
        marr = ManifestArray(zarray=zarray, chunkmanifest=chunkmanifest)
        # create encoding dict (and remove those keys from attrs)
        encoding_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}
        encoding = {key: value for key, value in attrs.items() if key in encoding_keys}
        attrs = {key: value for key, value in attrs.items() if key not in encoding_keys}
        return xr.Variable(dims=dims, data=marr, attrs=attrs, encoding=encoding)

    def _parse_attribute(self, attr_tag: ET.Element) -> dict:
        """
        Parse an attribute from a DMR attr tag.

        Parameters
        ----------
        attr_tag : ET.Element
            An ElementTree Element with an <Attr> tag.

        Returns
        -------
        dict
        """
        attr = {}
        values = []
        # if multiple Value tags are present, store as "key": "[v1, v2, ...]"
        for value_tag in attr_tag:
            values.append(value_tag.text)
        attr[attr_tag.attrib["name"]] = values[0] if len(values) == 1 else str(values)
        return attr

    def _parse_chunks(self, chunks_tag: ET.Element, chunks: tuple) -> dict:
        """
        Parse the chunk manifest from a DMR chunks tag.

        Parameters
        ----------
        chunks_tag : ET.Element
            An ElementTree Element with a <chunks> tag.

        chunks : tuple
            Chunk sizes for each dimension.

        Returns
        -------
        dict
        """
        chunkmanifest: dict[ChunkKey, object] = {}
        default_num: list[int] = [0 for i in range(len(chunks))]
        chunk_key_template = ".".join(["{}" for i in range(len(chunks))])
        for chunk_tag in chunks_tag.iterfind("dmr:chunk", self._ns):
            chunk_num = default_num
            if "chunkPositionInArray" in chunk_tag.attrib:
                # "[0,1023,10235]"" -> ["0","1023","10235"]
                chunk_pos = chunk_tag.attrib["chunkPositionInArray"][1:-1].split(",")
                # [0,1023,10235] // [1, 1023, 2047] -> [0,1,5]
                chunk_num = [int(chunk_pos[i]) // chunks[i] for i in range(len(chunks))]
            chunk_key = ChunkKey(
                chunk_key_template.format(*chunk_num)
            )  # [0,1,5] -> "0.1.5"
            chunkmanifest[chunk_key] = {
                "path": self.data_filepath,
                "offset": int(chunk_tag.attrib["offset"]),
                "length": int(chunk_tag.attrib["nBytes"]),
            }
        validate_chunk_keys(chunkmanifest.keys())
        return chunkmanifest
