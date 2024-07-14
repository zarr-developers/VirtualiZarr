import os
import warnings
from collections import defaultdict
from typing import Any, Optional
from xml.etree import ElementTree as ET

import numpy as np
import xarray as xr

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.types import ChunkKey
from virtualizarr.zarr import ZArray


class DMRParser:
    """
    Parses a DMR++ file and creates a virtual xr.Dataset.
    Handles groups, dimensions, coordinates, data variables, encoding, chunk manifests, and attributes.
    """

    # DAP and DMRPP XML namespaces
    _ns = {
        "dap": "http://xml.opendap.org/ns/DAP/4.0#",
        "dmr": "http://xml.opendap.org/dap/dmrpp/1.0.0#",
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
        "Url": "str",
        "Float32": "float32",
        "Float64": "float64",
        "String": "str",
    }
    # Default zlib compression value (-1 means default, currently level 6 is default)
    _default_zlib_value = -1
    # Encoding keys that should be cast to float
    _encoding_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

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

    def parse_dataset(self, group=None) -> xr.Dataset:
        """
        Parse the dataset from the dmrpp file

        Parameters
        ----------
        group : str
            The group to parse. If None, and no groups are present, the dataset is parsed.
            If None and groups are present, the first group is parsed.

        Returns
        -------
        An xr.Dataset wrapping virtualized zarr arrays.
        """
        if group is not None:
            # group = "/" + group.strip("/")  # ensure group is in form "/a/b"
            group = os.path.normpath(group).removeprefix(
                "/"
            )  # ensure group is in form "a/b/c"
        if self.data_filepath.endswith(".h5"):
            return self._parse_hdf5_dataset(self.root, group)
        if self.data_filepath.endswith(".nc"):
            return self._parse_netcdf4_dataset(self.root, group)
        raise ValueError("DMR file must be HDF5 or netCDF4 based")

    def _parse_netcdf4_dataset(
        self, root: ET.Element, group: Optional[str] = None
    ) -> xr.Dataset:
        """
        Parse the dataset from the netcdf4 based dmrpp with groups, starting at the given group.
        Set root to the given group.

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR file.

        group : str
            The group to parse. If None, and no groups are present, the dataset is parsed.
            If None and groups are present, the first group is parsed.
        Returns
        -------
        xr.Dataset
        """
        group_tags = root.findall("dap:Group", self._ns)
        if len(group_tags) == 0:
            if group is not None:
                # no groups found and group specified -> warning
                warnings.warn(
                    "No groups found in NetCDF4 DMR file; ignoring group parameter"
                )
            # no groups found and no group specified -> parse dataset
            return self._parse_dataset(root)
        all_groups = self._split_netcdf4(root)
        if group is None:
            # groups found and no group specified -> parse first group
            return self._parse_dataset(group_tags[0])
        if group in all_groups:
            # groups found and group specified -> parse specified group
            return self._parse_dataset(all_groups[group])
        else:
            # groups found and specified group not found -> error
            raise ValueError(f"Group {group} not found in NetCDF4 DMR file")

    def _split_netcdf4(self, root: ET.Element) -> dict[str, ET.Element]:
        """
        Split the input <Group> element into several <Dataset> ET.Elements by netcdf4 group
        E.g. {"left": <Dataset>, "right": <Dataset>}

        Returns
        -------
        dict[str, ET.Element]
        """
        group_tags = root.findall("dap:Group", self._ns)
        all_groups: dict[str, ET.Element] = defaultdict(
            lambda: ET.Element(root.tag, root.attrib)
        )
        for group_tag in group_tags:
            all_groups[os.path.normpath(group_tag.attrib["name"])] = group_tag
        return all_groups

    def _parse_hdf5_dataset(self, root: ET.Element, group: str) -> xr.Dataset:
        """
        Parse the dataset from the HDF5 based dmrpp with groups, starting at the given group.
        Set root to the given group.

        Parameters
        ----------
        root : ET.Element
            The root element of the DMR file.

        group : str
            The group to parse. If None, and no groups are present, the dataset is parsed.
            If None and groups are present, the first group is parsed.

        Returns
        -------
        xr.Dataset
        """
        all_groups = self._split_hdf5(root=root)
        if group in all_groups:
            # replace aliased variable names with original names: gt1r_heights -> heights
            orignames = {}
            vars_tags: list[ET.Element] = []
            for dap_dtype in self._dap_np_dtype:
                vars_tags += all_groups[group].findall(f"dap:{dap_dtype}", self._ns)
            for var_tag in vars_tags:
                origname_tag = var_tag.find(
                    "./dap:Attribute[@name='origname']/dap:Value", self._ns
                )
                if origname_tag is not None and origname_tag.text is not None:
                    orignames[var_tag.attrib["name"]] = origname_tag.text
            return self._parse_dataset(all_groups[group]).rename(orignames)
        raise ValueError(f"Group {group} not found in HDF5 DMR file")

    def _split_hdf5(self, root: ET.Element) -> dict[str, ET.Element]:
        """
        Split the input <Dataset> element into several <Dataset> ET.Elements by HDF5 group
        E.g. {"gtr1/heights": <Dataset>, "gtr1/temperatures": <Dataset>}

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
        group_dims: dict[str, set[str]] = defaultdict(
            set
        )  # {"gt1r/heights": {"dim1", "dim2", ...}}
        vars_tags: list[ET.Element] = []
        for dap_dtype in self._dap_np_dtype:
            vars_tags += root.findall(f"dap:{dap_dtype}", self._ns)
        # Variables
        for var_tag in vars_tags:
            fullname_tag = var_tag.find(
                "./dap:Attribute[@name='fullnamepath']/dap:Value", self._ns
            )
            if fullname_tag is not None and fullname_tag.text is not None:
                # '/gt1r/heights/ph_id_pulse' -> 'gt1r/heights'
                group_name = os.path.dirname(fullname_tag.text).removeprefix("/")
                all_groups[group_name].append(var_tag)
                for dim_tag in var_tag.iterfind("dap:Dim", self._ns):
                    group_dims[group_name].add(dim_tag.attrib["name"].removeprefix("/"))
        # Dimensions
        for dim_tag in root.iterfind("dap:Dimension", self._ns):
            for group_name, dims in group_dims.items():
                if dim_tag.attrib["name"] in dims:
                    all_groups[group_name].append(dim_tag)
        # Attributes
        for attr_tag in root.iterfind("dap:Attribute", self._ns):
            fullname_tag = attr_tag.find(
                "./dap:Attribute[@name='fullnamepath']/dap:Value", self._ns
            )
            if fullname_tag is not None and fullname_tag.text is not None:
                group_name = fullname_tag.text
                # Add all attributes to the new dataset; fullnamepath is generally excluded
                if group_name in all_groups:
                    all_groups[group_name].extend(
                        [a for a in attr_tag if a.attrib["name"] != "fullnamepath"]
                    )
        return all_groups

    def _parse_dataset(self, root: ET.Element) -> xr.Dataset:
        """
        Parse the dataset using the root element of the DMR file.

        Returns
        -------
        xr.Dataset
        """
        # Dimension names and sizes
        dataset_dims: dict[str, int] = {}
        for dim_tag in root.iterfind("dap:Dimension", self._ns):
            dataset_dims[dim_tag.attrib["name"]] = int(dim_tag.attrib["size"])
        # Data variables and coordinates
        vars_tags: list[ET.Element] = []
        for dap_dtype in self._dap_np_dtype:
            vars_tags += root.findall(f"dap:{dap_dtype}", self._ns)
        # Coordinate names (using Map tags and coordinates attribute)
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
        if len(coord_names) == 0 or len(coord_names) < len(dataset_dims):
            coord_names = set(dataset_dims.keys())
        # Seperate and parse coords + data variables
        coords: dict[str, xr.Variable] = {}
        data_vars: dict[str, xr.Variable] = {}
        for var_tag in vars_tags:
            if var_tag.attrib["name"] in coord_names:
                coords[var_tag.attrib["name"]] = self._parse_variable(
                    var_tag, dataset_dims
                )
            else:
                data_vars[var_tag.attrib["name"]] = self._parse_variable(
                    var_tag, dataset_dims
                )
        # Attributes
        attrs: dict[str, str] = {}
        for attr_tag in self.root.iterfind("dap:Attribute", self._ns):
            if attr_tag.attrib["type"] != "Container":  # container = nested attributes
                attrs.update(self._parse_attribute(attr_tag))
        return xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates(coords=coords, indexes={}),
            attrs=attrs,
        )

    def _parse_variable(
        self, var_tag: ET.Element, dataset_dims: dict[str, int]
    ) -> xr.Variable:
        """
        Parse a variable from a DMR tag.

        Parameters
        ----------
        var_tag : ET.Element
            An ElementTree Element representing a variable in the DMR file. Will have DAP dtype as tag.

        dataset_dims : dict
            A dictionary of dimension names and sizes. E.g. {"time": 1, "lat": 1447, "lon": 2895}
            Must contain at least all the dimensions used by the variable.

        Returns
        -------
        xr.Variable
        """
        # Dimension names
        dim_names: list[str] = []
        for dim_tag in var_tag.iterfind("dap:Dim", self._ns):
            dim_names.append(os.path.basename(dim_tag.attrib["name"]))
        # convert DAP dtype to numpy dtype
        dtype = np.dtype(
            self._dap_np_dtype[var_tag.tag.removeprefix("{" + self._ns["dap"] + "}")]
        )
        # Chunks and Filters
        filters = None
        fill_value = np.nan
        shape = tuple([dataset_dims[d] for d in dim_names])
        chunks_shape = shape
        chunks_tag = var_tag.find("dmr:chunks", self._ns)
        if chunks_tag is not None:
            # Chunks
            chunk_dim_tag = chunks_tag.find("dmr:chunkDimensionSizes", self._ns)
            if chunk_dim_tag is not None and chunk_dim_tag.text is not None:
                # 1 1447 2895 -> (1, 1447, 2895)
                chunks_shape = tuple(map(int, chunk_dim_tag.text.split()))
            chunkmanifest = self._parse_chunks(chunks_tag, chunks_shape)
            # Filters
            if "compressionType" in chunks_tag.attrib:
                filters = []
                # shuffle deflate --> ["shuffle", "deflate"]
                compression_types = chunks_tag.attrib["compressionType"].split(" ")
                for c in compression_types:
                    if c == "shuffle":
                        filters.append({"id": "shuffle", "elementsize": dtype.itemsize})
                    elif c == "deflate":
                        filters.append(
                            {"id": "zlib", "level": self._default_zlib_value}
                        )
        # Attributes
        attrs: dict[str, Any] = {}
        for attr_tag in var_tag.iterfind("dap:Attribute", self._ns):
            attrs.update(self._parse_attribute(attr_tag))
        if "_FillValue" in attrs and attrs["_FillValue"] != "*":
            fill_value = attrs["_FillValue"]
        attrs.pop("_FillValue", None)
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
        return xr.Variable(dims=dim_names, data=marr, attrs=attrs, encoding=encoding)

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
        dtype = np.dtype(self._dap_np_dtype[attr_tag.attrib["type"]])
        # if multiple Value tags are present, store as "key": "[v1, v2, ...]"
        for value_tag in attr_tag:
            # cast attribute to native python type using dmr provided dtype
            values.append(dtype.type(value_tag.text).item())
        attr[attr_tag.attrib["name"]] = values[0] if len(values) == 1 else values
        return attr

    def _parse_chunks(
        self, chunks_tag: ET.Element, chunks_shape: tuple[int, ...]
    ) -> ChunkManifest:
        """
        Parse the chunk manifest from a DMR chunks tag.

        Parameters
        ----------
        chunks_tag : ET.Element
            An ElementTree Element with a <chunks> tag.

        chunks : tuple
            Chunk sizes for each dimension. E.g. (1, 1447, 2895)

        Returns
        -------
        ChunkManifest
        """
        chunkmanifest: dict[ChunkKey, object] = {}
        default_num: list[int] = [0 for i in range(len(chunks_shape))]
        chunk_key_template = ".".join(["{}" for i in range(len(chunks_shape))])
        for chunk_tag in chunks_tag.iterfind("dmr:chunk", self._ns):
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
