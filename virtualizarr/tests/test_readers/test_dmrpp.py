from xml.etree import ElementTree as ET

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.readers.dmrpp import DMRParser
from virtualizarr.tests import network
from VirtualiZarr.virtualizarr.manifests.manifest import ChunkManifest

urls = [
    (
        "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "https://its-live-data.s3-us-west-2.amazonaws.com/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
    )
    # TODO: later add MUR, SWOT, TEMPO and others by using kerchunk JSON to read refs (rather than reading the whole netcdf file)
]


@pytest.fixture
def basic_dmrpp() -> DMRParser:
    xml_str = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" dapVersion="4.0" dmrVersion="1.0" name="test.dmrpp">
        <Dimension name="x" size="720"/>
        <Dimension name="y" size="1440"/>
        <Dimension name="z" size="3"/>
        <Int32 name="x">
            <Dim name="/x"/>
            <Attribute name="long_name" type="String">
                <Value>grid x-axis</Value>
            </Attribute>
            <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41268" nBytes="4"/>
            </dmrpp:chunks>
        </Int32>
        <Int32 name="y">
            <Dim name="/y"/>
            <Attribute name="long_name" type="String">
                <Value>grid y-axis</Value>
            </Attribute>
            <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41272" nBytes="4"/>
            </dmrpp:chunks>
        </Int32>
        <Int32 name="z">
            <Dim name="/z"/>
            <Attribute name="long_name" type="String">
                <Value>grid z-axis</Value>
            </Attribute>
            <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41276" nBytes="4"/>
            </dmrpp:chunks>
        </Int32>
        <Float32 name="data">
            <Dim name="/x"/>
            <Dim name="/y"/>
            <Attribute name="long_name" type="String">
                <Value>analysed sea surface temperature</Value>
            </Attribute>
            <Attribute name="items" type="Int16">
                <Value>1</Value>
                <Value>2</Value>
                <Value>3</Value>
            </Attribute>
            <Attribute name="_FillValue" type="Int16">
                <Value>-32768</Value>
            </Attribute>
            <Attribute name="add_offset" type="Float64">
                <Value>298.14999999999998</Value>
            </Attribute>
            <Attribute name="scale_factor" type="Float64">
                <Value>0.001</Value>
            </Attribute>
            <Attribute name="coordinates" type="String">
                <Value>x y z</Value>
            </Attribute>
            <dmrpp:chunks compressionType="shuffle deflate" deflateLevel="5" fillValue="-32768" byteOrder="LE">
                <dmrpp:chunkDimensionSizes>360 720</dmrpp:chunkDimensionSizes>
                <dmrpp:chunk offset="40762" nBytes="4083" chunkPositionInArray="[0,0]"/>
                <dmrpp:chunk offset="44845" nBytes="4083" chunkPositionInArray="[0,720]"/>
                <dmrpp:chunk offset="48928" nBytes="4083" chunkPositionInArray="[0,1440]"/>
                <dmrpp:chunk offset="53011" nBytes="4083" chunkPositionInArray="[360, 0]"/>
                <dmrpp:chunk offset="57094" nBytes="4083" chunkPositionInArray="[360, 720]"/>
                <dmrpp:chunk offset="61177" nBytes="4083" chunkPositionInArray="[360, 1440]"/>
                <dmrpp:chunk offset="65260" nBytes="4083" chunkPositionInArray="[720, 0]"/>
                <dmrpp:chunk offset="69343" nBytes="4083" chunkPositionInArray="[720, 720]"/>
                <dmrpp:chunk offset="73426" nBytes="4083" chunkPositionInArray="[720, 1440]"/>
            </dmrpp:chunks>
        </Float32>
        <Float32 name="mask">
            <Dim name="/x"/>
            <Dim name="/y"/>
            <Dim name="/z"/>
            <Attribute name="long_name" type="String">
                <Value>mask</Value>
            </Attribute>
            <dmrpp:chunks compressionType="shuffle" fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41276" nBytes="4"/>
            </dmrpp:chunks>
        </Float32>
        <Attribute name="Conventions" type="String">
            <Value>CF-1.6</Value>
        </Attribute>
        <Attribute name="title" type="String">
            <Value>Sample Dataset</Value>
        </Attribute>
    </Dataset>
    """
    return DMRParser(root=ET.fromstring(xml_str))


@pytest.fixture
def nested_groups_dmrpp() -> DMRParser:
    xml_str = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" dapVersion="4.0" dmrVersion="1.0" name="test.dmrpp">
        <Dimension name="a" size="10"/>
        <Dimension name="b" size="10"/>
        <Int32 name="a">
            <Dim name="/a"/>
            <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41268" nBytes="4"/>
            </dmrpp:chunks>
        </Int32>
        <Int32 name="b">
            <Dim name="/b"/>
            <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                <dmrpp:chunk offset="41268" nBytes="4"/>
            </dmrpp:chunks>
        </Int32>
        <Group name="group1">
            <Dimension name="x" size="720"/>
            <Dimension name="y" size="1440"/>
            <Int32 name="x">
                <Dim name="/group1/x"/>
                <Attribute name="test" type="String">
                    <Value>test</Value>
                </Attribute>
                <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                    <dmrpp:chunk offset="41268" nBytes="4"/>
                </dmrpp:chunks>
            </Int32>
            <Int32 name="y">
                <Dim name="/group1/y"/>
                <Attribute name="test" type="String">
                    <Value>test</Value>
                </Attribute>
                <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                    <dmrpp:chunk offset="41268" nBytes="4"/>
                </dmrpp:chunks>
            </Int32>
            <Group name="group2">
                <Int32 name="area">
                    <Dim name="/group1/x"/>
                    <Dim name="/group1/y"/>
                    <dmrpp:chunks fillValue="-2147483647" byteOrder="LE">
                        <dmrpp:chunk offset="41268" nBytes="4"/>
                    </dmrpp:chunks>
                </Int32>
            </Group>
        </Group>
    </Dataset>
    """
    return DMRParser(root=ET.fromstring(xml_str))


@network
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_NASA_dmrpp(data_url, dmrpp_url):
    result = open_virtual_dataset(dmrpp_url, indexes={}, filetype="dmrpp")
    expected = open_virtual_dataset(data_url, indexes={})
    xr.testing.assert_identical(result, expected)


@pytest.mark.parametrize(
    "dmrpp_fixture, fqn_path, expected_xpath",
    [
        ("basic_dmrpp", "/", "."),
        ("basic_dmrpp", "/data", "./*[@name='data']"),
        ("basic_dmrpp", "/data/items", "./*[@name='data']/*[@name='items']"),
        (
            "nested_groups_dmrpp",
            "/group1/group2/area",
            "./*[@name='group1']/*[@name='group2']/*[@name='area']",
        ),
    ],
)
def test_find_node_fqn(request, dmrpp_fixture, fqn_path, expected_xpath):
    parser_instance = request.getfixturevalue(dmrpp_fixture)
    result = parser_instance.find_node_fqn(fqn_path)
    expected = parser_instance.root.find(expected_xpath, parser_instance._NS)
    assert result == expected


@pytest.mark.parametrize(
    "dmrpp_fixture, group_path",
    [
        ("basic_dmrpp", "/"),
        ("nested_groups_dmrpp", "/"),
        ("nested_groups_dmrpp", "/group1"),
        ("nested_groups_dmrpp", "/group1/group2"),
    ],
)
def test_split_groups(request, dmrpp_fixture, group_path):
    dmrpp_instance = request.getfixturevalue(dmrpp_fixture)
    # get all tags in a dataset (so all tags excluding nested groups)
    dataset_tags = lambda x: [
        d for d in x if d.tag != "{" + dmrpp_instance._NS["dap"] + "}" + "Group"
    ]
    # check that contents of the split groups dataset match contents of the original dataset
    result_tags = dataset_tags(
        dmrpp_instance._split_groups(dmrpp_instance.root)[group_path]
    )
    expected_tags = dataset_tags(dmrpp_instance.find_node_fqn(group_path))
    assert result_tags == expected_tags


def test_parse_dataset(basic_dmrpp, nested_groups_dmrpp):
    vds = basic_dmrpp.parse_dataset()
    assert vds.sizes == {"x": 720, "y": 1440, "z": 3}
    assert vds.data_vars.keys() == {"data", "mask"}
    assert vds.data_vars["data"].dims == ("x", "y")
    assert vds.attrs == {"Conventions": "CF-1.6", "title": "Sample Dataset"}
    assert vds.coords.keys() == {"x", "y", "z"}
    vds_root_implicit = nested_groups_dmrpp.parse_dataset()
    vds_root = nested_groups_dmrpp.parse_dataset(group="/")
    xrt.assert_identical(vds_root_implicit, vds_root)
    assert vds_root.sizes == {"a": 10, "b": 10}
    assert vds_root.coords.keys() == {"a", "b"}
    vds_g1 = nested_groups_dmrpp.parse_dataset(group="/group1")
    assert vds_g1.sizes == {"x": 720, "y": 1440}
    assert vds_g1.coords.keys() == {"x", "y"}
    vds_g2 = nested_groups_dmrpp.parse_dataset(group="/group1/group2")
    assert vds_g2.sizes == {"x": 720, "y": 1440}
    assert vds_g2.data_vars.keys() == {"area"}
    assert vds_g2.data_vars["area"].dims == ("x", "y")


@pytest.mark.parametrize(
    "dim_path, expected",
    [
        ("/a", {"a": 10}),
        ("/group1/x", {"x": 720}),
    ],
)
def test_parse_dim(nested_groups_dmrpp, dim_path, expected):
    result = nested_groups_dmrpp._parse_dim(nested_groups_dmrpp.find_node_fqn(dim_path))
    assert result == expected


@pytest.mark.parametrize("dim_path", ["/", "/mask"])
def test_find_dimension_tags(basic_dmrpp, dim_path):
    # Check that Dimension tags match Dimension tags from the root
    # Check that Dim tags reference the same Dimension tags from the root
    assert basic_dmrpp._find_dimension_tags(
        basic_dmrpp.find_node_fqn(dim_path)
    ) == basic_dmrpp.root.findall("dap:Dimension", basic_dmrpp._NS)


def test_parse_variable(basic_dmrpp):
    var = basic_dmrpp._parse_variable(basic_dmrpp.find_node_fqn("/data"))
    assert var.dtype == "float32"
    assert var.dims == ("x", "y")
    assert var.shape == (720, 1440)
    assert var.data.zarray.chunks == (360, 720)
    assert var.data.zarray.fill_value == -32768
    assert var.encoding == {"add_offset": 298.15, "scale_factor": 0.001}
    assert var.attrs == {
        "long_name": "analysed sea surface temperature",
        "items": [1, 2, 3],
        "coordinates": "x y z",
        "add_offset": 298.15,
        "scale_factor": 0.001,
    }


@pytest.mark.parametrize(
    "attr_path, expected",
    [
        ("data/long_name", {"long_name": "analysed sea surface temperature"}),
        ("data/items", {"items": [1, 2, 3]}),
        ("data/_FillValue", {"_FillValue": -32768}),
    ],
)
def test_parse_attribute(basic_dmrpp, attr_path, expected):
    result = basic_dmrpp._parse_attribute(basic_dmrpp.find_node_fqn(attr_path))
    assert result == expected


@pytest.mark.parametrize(
    "var_path, dtype, expected_filters",
    [
        (
            "/data",
            np.dtype("float32"),
            [
                {"elementsize": np.dtype("float32").itemsize, "id": "shuffle"},
                {"id": "zlib", "level": 5},
            ],
        ),
        (
            "/mask",
            np.dtype("float32"),
            [{"elementsize": np.dtype("float32").itemsize, "id": "shuffle"}],
        ),
    ],
)
def test_parse_filters(basic_dmrpp, var_path, dtype, expected_filters):
    chunks_tag = basic_dmrpp.find_node_fqn(var_path).find(
        "dmrpp:chunks", basic_dmrpp._NS
    )
    result = basic_dmrpp._parse_filters(chunks_tag, dtype)
    assert result == expected_filters


@pytest.mark.parametrize(
    "var_path, chunk_shape, expected_lengths, expected_offsets, expected_paths",
    [
        (
            "/data",
            (360, 720),
            np.full((3, 3), 4083, dtype=np.uint64),
            (np.arange(9, dtype=np.uint64) * 4083 + 40762).reshape(3, 3),
            np.full((3, 3), "test.dmrpp", dtype=np.dtypes.StringDType),
        ),
        (
            "/mask",
            (720, 1440),
            np.array([4], dtype=np.uint64),
            np.array([41276], dtype=np.uint64),
            np.array(["test.dmrpp"], dtype=np.dtypes.StringDType),
        ),
    ],
)
def test_parse_chunks(
    basic_dmrpp,
    var_path,
    chunk_shape,
    expected_lengths,
    expected_offsets,
    expected_paths,
):
    chunks_tag = basic_dmrpp.find_node_fqn(var_path).find(
        "dmrpp:chunks", basic_dmrpp._NS
    )
    result = basic_dmrpp._parse_chunks(chunks_tag, chunk_shape)
    expected = ChunkManifest.from_arrays(
        lengths=expected_lengths, offsets=expected_offsets, paths=expected_paths
    )
    assert result == expected
