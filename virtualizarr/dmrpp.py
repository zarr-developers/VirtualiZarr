from xml.etree import ElementTree as ET

import numpy as np
import xarray as xr

from virtualizarr.manifests import ManifestArray
from virtualizarr.zarr import ZArray


class DMRParser:
    dap_namespace = "{http://xml.opendap.org/ns/DAP/4.0#}"
    dmr_namespace = "{http://xml.opendap.org/dap/dmrpp/1.0.0#}"
    dap_npdtype = {
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

    def __init__(self, dmr: str):
        self.root = ET.fromstring(dmr)
        self.data_filepath = self.root.attrib["name"]
        self.global_dims = {}

    def parse_dataset(self):
        # find all dimension names and sizes
        for d in self.root.iterfind(self.dap_namespace + "Dimension"):
            self.global_dims[d.attrib["name"]] = int(d.attrib["size"])
        vars_tags = []
        for dap_dtype in self.dap_npdtype:
            vars_tags += self.root.findall(self.dap_namespace + dap_dtype)
        # find all coordinate names (using Map tags)
        coord_names = set()
        for var_tag in vars_tags:
            for map_tag in var_tag.iterfind(self.dap_namespace + "Map"):
                coord_names.add(map_tag.attrib["name"].removeprefix("/"))
        coords = {}
        data_vars = {}
        for var_tag in vars_tags:
            if var_tag.attrib["name"] in coord_names:
                coords[var_tag.attrib["name"]] = self.parse_variable(var_tag)
                # if len(coords[v.attrib['name']].dims) == 1:
                #     dim1d, *_ = coords[v.attrib['name']].dims
                #     indexes[v.attrib['name']] = PandasIndex(coords[v.attrib['name']], dim1d)
            else:
                data_vars[var_tag.attrib["name"]] = self.parse_variable(var_tag)
        # find all dataset attributes
        attrs = {}
        for attr_tag in self.root.iterfind(self.dap_namespace + "Attribute"):
            if attr_tag.attrib["type"] != "Container":
                attrs.update(self.parse_attribute(attr_tag))
        return xr.Dataset(
            data_vars=data_vars,
            coords=xr.Coordinates(coords=coords, indexes={}),
            attrs=attrs,
        )

    def parse_variable(self, root) -> xr.Variable:
        # parse dimensions
        dims = []
        for d in root.iterfind(self.dap_namespace + "Dim"):
            dims.append(d.attrib["name"].removeprefix("/"))
        shape = tuple([self.global_dims[d] for d in dims])
        # parse chunks
        chunks = shape
        chunks_tag = root.find(self.dmr_namespace + "chunks")
        if chunks_tag.find(self.dmr_namespace + "chunkDimensionSizes") is not None:
            dim_str = chunks_tag.find(self.dmr_namespace + "chunkDimensionSizes").text
            chunks = tuple(map(int, dim_str.split()))
        chunkmanifest = self.parse_chunks(chunks_tag, chunks)
        # parse attributes
        attrs = {}
        for a in root.iterfind(self.dap_namespace + "Attribute"):
            attrs.update(self.parse_attribute(a))
        # create ManifestArray and ZArray
        dtype = np.dtype(self.dap_npdtype[root.tag.removeprefix(self.dap_namespace)])
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

    def parse_attribute(self, root) -> dict:
        attr = {}
        values = []
        # if multiple Value tags are present, store as "key": "[v1, v2, ...]"
        for r in root:
            values.append(r.text)
        attr[root.attrib["name"]] = values[0] if len(values) == 1 else str(values)
        return attr

    def parse_chunks(self, root, chunks: tuple) -> dict:
        chunkmanifest = {}
        default_num = [0 for i in range(len(chunks))]
        chunk_key_template = ".".join(["{}" for i in range(len(chunks))])
        for r in root.iterfind(self.dmr_namespace + "chunk"):
            chunk_num = default_num
            if "chunkPositionInArray" in r.attrib:
                # [0,1023,10235] // [1, 1023, 2047] -> [0,1,5]
                chunk_pos = r.attrib["chunkPositionInArray"][1:-1].split(",")
                chunk_num = [int(chunk_pos[i]) // chunks[i]
                             for i in range(len(chunks))]
            # [0,0,1] -> "0.0.1"
            chunk_key = chunk_key_template.format(*chunk_num)
            chunkmanifest[chunk_key] = {
                "path": self.data_filepath,
                "offset": int(r.attrib["offset"]),
                "length": int(r.attrib["nBytes"]),
            }
        return chunkmanifest
