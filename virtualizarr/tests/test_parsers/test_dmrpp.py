import textwrap
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import xarray as xr
import xarray.testing as xrt
from packaging import version

from virtualizarr.parsers import DMRPPParser, HDFParser
from virtualizarr.parsers.dmrpp import DMRParser
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import requires_network, slow_test
from virtualizarr.tests.utils import obstore_local, obstore_s3
from virtualizarr.xarray import open_virtual_dataset

urls = [
    (
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
    )
    # TODO: later add MUR, SWOT, TEMPO and others by using kerchunk JSON to read refs (rather than reading the whole netcdf file)
]

# This DMRPP was created following the instructions from https://opendap.github.io/DMRpp-wiki/DMRpp.html#sec-build-them on the files produced by conftest.py with the key matching the fixture name.
DMRPP_XML_STRINGS = {
    "netcdf4_file": textwrap.dedent(
        """\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" dapVersion="4.0" dmrVersion="1.0" name="air.nc" dmrpp:href="OPeNDAP_DMRpp_DATA_ACCESS_URL" dmrpp:version="3.21.1-451">
            <Dimension name="lat" size="25"/>
            <Dimension name="lon" size="53"/>
            <Dimension name="time" size="2920"/>
            <Float32 name="lat">
                <Dim name="/lat"/>
                <Attribute name="_FillValue" type="Float32">
                    <Value>NaN</Value>
                </Attribute>
                <Attribute name="standard_name" type="String">
                    <Value>latitude</Value>
                </Attribute>
                <Attribute name="long_name" type="String">
                    <Value>Latitude</Value>
                </Attribute>
                <Attribute name="units" type="String">
                    <Value>degrees_north</Value>
                </Attribute>
                <Attribute name="axis" type="String">
                    <Value>Y</Value>
                </Attribute>
                <dmrpp:chunks fillValue="nan" byteOrder="LE">
                    <dmrpp:chunk offset="5179" nBytes="100"/>
                </dmrpp:chunks>
            </Float32>
            <Float32 name="lon">
                <Dim name="/lon"/>
                <Attribute name="_FillValue" type="Float32">
                    <Value>NaN</Value>
                </Attribute>
                <Attribute name="standard_name" type="String">
                    <Value>longitude</Value>
                </Attribute>
                <Attribute name="long_name" type="String">
                    <Value>Longitude</Value>
                </Attribute>
                <Attribute name="units" type="String">
                    <Value>degrees_east</Value>
                </Attribute>
                <Attribute name="axis" type="String">
                    <Value>X</Value>
                </Attribute>
                <dmrpp:chunks fillValue="nan" byteOrder="LE">
                    <dmrpp:chunk offset="5279" nBytes="212"/>
                </dmrpp:chunks>
            </Float32>
            <Float32 name="time">
                <Dim name="/time"/>
                <Attribute name="_FillValue" type="Float32">
                    <Value>NaN</Value>
                </Attribute>
                <Attribute name="standard_name" type="String">
                    <Value>time</Value>
                </Attribute>
                <Attribute name="long_name" type="String">
                    <Value>Time</Value>
                </Attribute>
                <Attribute name="units" type="String">
                    <Value>hours since 1800-01-01</Value>
                </Attribute>
                <Attribute name="calendar" type="String">
                    <Value>standard</Value>
                </Attribute>
                <dmrpp:chunks fillValue="nan" byteOrder="LE">
                    <dmrpp:chunk offset="7757499" nBytes="11680"/>
                </dmrpp:chunks>
            </Float32>
            <Int16 name="air">
                <Dim name="/time"/>
                <Dim name="/lat"/>
                <Dim name="/lon"/>
                <Attribute name="long_name" type="String">
                    <Value>4xDaily Air temperature at sigma level 995</Value>
                </Attribute>
                <Attribute name="units" type="String">
                    <Value>degK</Value>
                </Attribute>
                <Attribute name="precision" type="Int16">
                    <Value>2</Value>
                </Attribute>
                <Attribute name="GRIB_id" type="Int16">
                    <Value>11</Value>
                </Attribute>
                <Attribute name="GRIB_name" type="String">
                    <Value>TMP</Value>
                </Attribute>
                <Attribute name="var_desc" type="String">
                    <Value>Air temperature</Value>
                </Attribute>
                <Attribute name="dataset" type="String">
                    <Value>NMC Reanalysis</Value>
                </Attribute>
                <Attribute name="level_desc" type="String">
                    <Value>Surface</Value>
                </Attribute>
                <Attribute name="statistic" type="String">
                    <Value>Individual Obs</Value>
                </Attribute>
                <Attribute name="parent_stat" type="String">
                    <Value>Other</Value>
                </Attribute>
                <Attribute name="actual_range" type="Float32">
                    <Value>185.1600037</Value>
                    <Value>322.1000061</Value>
                </Attribute>
                <Attribute name="scale_factor" type="Float64">
                    <Value>0.01</Value>
                </Attribute>
                <Map name="/time"/>
                <Map name="/lat"/>
                <Map name="/lon"/>
                <dmrpp:chunks fillValue="-32767" byteOrder="LE">
                    <dmrpp:chunk offset="10283" nBytes="7738000"/>
                </dmrpp:chunks>
            </Int16>
            <Attribute name="Conventions" type="String">
                <Value>COARDS</Value>
            </Attribute>
            <Attribute name="title" type="String">
                <Value>4x daily NMC reanalysis (1948)</Value>
            </Attribute>
            <Attribute name="description" type="String">
                <Value>Data is from NMC initialized reanalysis
        (4x/day).  These are the 0.9950 sigma level values.</Value>
            </Attribute>
            <Attribute name="platform" type="String">
                <Value>Model</Value>
            </Attribute>
            <Attribute name="references" type="String">
                <Value>http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html</Value>
            </Attribute>
            <Attribute name="build_dmrpp_metadata" type="Container">
                <Attribute name="created" type="String">
                    <Value>2025-07-16T18:48:42Z</Value>
                </Attribute>
                <Attribute name="build_dmrpp" type="String">
                    <Value>3.21.1-451</Value>
                </Attribute>
                <Attribute name="bes" type="String">
                    <Value>3.21.1-451</Value>
                </Attribute>
                <Attribute name="libdap" type="String">
                    <Value>libdap-3.21.1-178</Value>
                </Attribute>
                <Attribute name="invocation" type="String">
                    <Value>build_dmrpp -f /usr/share/hyrax/air.nc -r air.nc.dmr -u OPeNDAP_DMRpp_DATA_ACCESS_URL -M</Value>
                </Attribute>
            </Attribute>
        </Dataset>
        """
    ),
    "hdf5_groups_file": textwrap.dedent(
        """\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" dapVersion="4.0" dmrVersion="1.0" name="hdf5_groups_file.nc" dmrpp:href="OPeNDAP_DMRpp_DATA_ACCESS_URL" dmrpp:version="3.21.1-451">
            <Attribute name="build_dmrpp_metadata" type="Container">
                <Attribute name="created" type="String">
                    <Value>2025-07-16T21:57:57Z</Value>
                </Attribute>
                <Attribute name="build_dmrpp" type="String">
                    <Value>3.21.1-451</Value>
                </Attribute>
                <Attribute name="bes" type="String">
                    <Value>3.21.1-451</Value>
                </Attribute>
                <Attribute name="libdap" type="String">
                    <Value>libdap-3.21.1-178</Value>
                </Attribute>
                <Attribute name="invocation" type="String">
                    <Value>build_dmrpp -f /usr/share/hyrax/hdf5_groups_file.nc -r hdf5_groups_file.nc.dmr -u OPeNDAP_DMRpp_DATA_ACCESS_URL -M</Value>
                </Attribute>
            </Attribute>
            <Group name="test">
                <Group name="group">
                    <Dimension name="lat" size="25"/>
                    <Dimension name="lon" size="53"/>
                    <Dimension name="time" size="2920"/>
                    <Float32 name="lat">
                        <Dim name="/test/group/lat"/>
                        <Attribute name="_FillValue" type="Float32">
                            <Value>NaN</Value>
                        </Attribute>
                        <Attribute name="standard_name" type="String">
                            <Value>latitude</Value>
                        </Attribute>
                        <Attribute name="long_name" type="String">
                            <Value>Latitude</Value>
                        </Attribute>
                        <Attribute name="units" type="String">
                            <Value>degrees_north</Value>
                        </Attribute>
                        <Attribute name="axis" type="String">
                            <Value>Y</Value>
                        </Attribute>
                        <dmrpp:chunks fillValue="nan" byteOrder="LE">
                            <dmrpp:chunk offset="5533" nBytes="100"/>
                        </dmrpp:chunks>
                    </Float32>
                    <Float32 name="lon">
                        <Dim name="/test/group/lon"/>
                        <Attribute name="_FillValue" type="Float32">
                            <Value>NaN</Value>
                        </Attribute>
                        <Attribute name="standard_name" type="String">
                            <Value>longitude</Value>
                        </Attribute>
                        <Attribute name="long_name" type="String">
                            <Value>Longitude</Value>
                        </Attribute>
                        <Attribute name="units" type="String">
                            <Value>degrees_east</Value>
                        </Attribute>
                        <Attribute name="axis" type="String">
                            <Value>X</Value>
                        </Attribute>
                        <dmrpp:chunks fillValue="nan" byteOrder="LE">
                            <dmrpp:chunk offset="5633" nBytes="212"/>
                        </dmrpp:chunks>
                    </Float32>
                    <Float32 name="time">
                        <Dim name="/test/group/time"/>
                        <Attribute name="_FillValue" type="Float32">
                            <Value>NaN</Value>
                        </Attribute>
                        <Attribute name="standard_name" type="String">
                            <Value>time</Value>
                        </Attribute>
                        <Attribute name="long_name" type="String">
                            <Value>Time</Value>
                        </Attribute>
                        <Attribute name="units" type="String">
                            <Value>hours since 1800-01-01</Value>
                        </Attribute>
                        <Attribute name="calendar" type="String">
                            <Value>standard</Value>
                        </Attribute>
                        <dmrpp:chunks fillValue="nan" byteOrder="LE">
                            <dmrpp:chunk offset="7756034" nBytes="11680"/>
                        </dmrpp:chunks>
                    </Float32>
                    <Int16 name="air">
                        <Dim name="/test/group/time"/>
                        <Dim name="/test/group/lat"/>
                        <Dim name="/test/group/lon"/>
                        <Attribute name="long_name" type="String">
                            <Value>4xDaily Air temperature at sigma level 995</Value>
                        </Attribute>
                        <Attribute name="units" type="String">
                            <Value>degK</Value>
                        </Attribute>
                        <Attribute name="precision" type="Int16">
                            <Value>2</Value>
                        </Attribute>
                        <Attribute name="GRIB_id" type="Int16">
                            <Value>11</Value>
                        </Attribute>
                        <Attribute name="GRIB_name" type="String">
                            <Value>TMP</Value>
                        </Attribute>
                        <Attribute name="var_desc" type="String">
                            <Value>Air temperature</Value>
                        </Attribute>
                        <Attribute name="dataset" type="String">
                            <Value>NMC Reanalysis</Value>
                        </Attribute>
                        <Attribute name="level_desc" type="String">
                            <Value>Surface</Value>
                        </Attribute>
                        <Attribute name="statistic" type="String">
                            <Value>Individual Obs</Value>
                        </Attribute>
                        <Attribute name="parent_stat" type="String">
                            <Value>Other</Value>
                        </Attribute>
                        <Attribute name="actual_range" type="Float32">
                            <Value>185.1600037</Value>
                            <Value>322.1000061</Value>
                        </Attribute>
                        <Attribute name="scale_factor" type="Float64">
                            <Value>0.01</Value>
                        </Attribute>
                        <Map name="/test/group/time"/>
                        <Map name="/test/group/lat"/>
                        <Map name="/test/group/lon"/>
                        <dmrpp:chunks fillValue="-32767" byteOrder="LE">
                            <dmrpp:chunk offset="10554" nBytes="7738000"/>
                        </dmrpp:chunks>
                    </Int16>
                    <Attribute name="Conventions" type="String">
                        <Value>COARDS</Value>
                    </Attribute>
                    <Attribute name="title" type="String">
                        <Value>4x daily NMC reanalysis (1948)</Value>
                    </Attribute>
                    <Attribute name="description" type="String">
                        <Value>Data is from NMC initialized reanalysis
        (4x/day).  These are the 0.9950 sigma level values.</Value>
                    </Attribute>
                    <Attribute name="platform" type="String">
                        <Value>Model</Value>
                    </Attribute>
                    <Attribute name="references" type="String">
                        <Value>http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html</Value>
                    </Attribute>
                </Group>
            </Group>
        </Dataset>
            """
    ),
    "fill_value_scalar_no_chunks_nc4_url": textwrap.dedent(
        """\
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <Dataset xmlns="http://xml.opendap.org/ns/DAP/4.0#" xmlns:dmrpp="http://xml.opendap.org/dap/dmrpp/1.0.0#" dapVersion="4.0" dmrVersion="1.0" name="fill_value_scalar_no_chunks.nc4" dmrpp:href="OPeNDAP_DMRpp_DATA_ACCESS_URL" dmrpp:version="3.21.1-477">
            <Int32 name="data">
                <Attribute name="_FillValue" type="Int32">
                    <Value>-999</Value>
                </Attribute>
                <dmrpp:chunks fillValue="-999"/>
            </Int32>
            <Attribute name="long_name" type="String">
                <Value>empty scalar data</Value>
            </Attribute>
            <Attribute name="drop_container_attribute" type="Container">
                <Attribute name="created" type="String">
                    <Value>2025-08-14T23:32:01Z</Value>
                </Attribute>
                <Attribute name="reason" type="String">
                    <Value>container attributes are no longer supported</Value>
                </Attribute>
            </Attribute>
            <Attribute name="build_dmrpp_metadata" type="Container">
                <Attribute name="created" type="String">
                    <Value>2025-08-14T23:32:01Z</Value>
                </Attribute>
                <Attribute name="build_dmrpp" type="String">
                    <Value>3.21.1-477</Value>
                </Attribute>
                <Attribute name="bes" type="String">
                    <Value>3.21.1-477</Value>
                </Attribute>
                <Attribute name="libdap" type="String">
                    <Value>libdap-3.21.1-222</Value>
                </Attribute>
                <Attribute name="invocation" type="String">
                    <Value>build_dmrpp -f /usr/share/hyrax/fill_value_scalar_no_chunks.nc4 -r fill_value_scalar_no_chunks.nc4.dmr -u OPeNDAP_DMRpp_DATA_ACCESS_URL -M</Value>
                </Attribute>
            </Attribute>
        </Dataset>
        """
    ),
}


def dmrparser(dmrpp_xml_str: str, filepath: str) -> DMRParser:
    # TODO we should actually create a dmrpp file in a temporary directory
    # this would avoid the need to pass tmp_path separately

    return DMRParser(root=ET.fromstring(dmrpp_xml_str), data_filepath=filepath)


@slow_test
@requires_network
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_NASA_dmrpp(data_url, dmrpp_url):
    store = obstore_s3(
        url=dmrpp_url,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry()
    registry.register("s3://its-live-data/test-space", store)
    with (
        open_virtual_dataset(
            url=dmrpp_url,
            registry=registry,
            parser=DMRPPParser(),
            loadable_variables=[],
        ) as actual,
        open_virtual_dataset(
            url=data_url,
            registry=registry,
            parser=HDFParser(),
            loadable_variables=[],
        ) as expected,
    ):
        xr.testing.assert_identical(actual, expected)


@requires_network
@slow_test
@pytest.mark.parametrize("data_url, dmrpp_url", urls)
def test_NASA_dmrpp_load(data_url, dmrpp_url):
    store = obstore_s3(
        url=dmrpp_url,
        region="us-west-2",
    )
    registry = ObjectStoreRegistry()
    registry.register(dmrpp_url, store)
    parser = DMRPPParser()
    manifest_store = parser(url=dmrpp_url, registry=registry)

    with xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    ) as ds:
        assert ds.load()


@pytest.mark.parametrize(
    "fqn_path, expected_xpath",
    [
        ("/", "."),
        ("/air", "./*[@name='air']"),
    ],
)
def test_find_node_fqn_simple(netcdf4_file, fqn_path, expected_xpath):
    parser_instance = dmrparser(
        DMRPP_XML_STRINGS["netcdf4_file"], filepath=netcdf4_file
    )
    result = parser_instance.find_node_fqn(fqn_path)
    expected = parser_instance.root.find(expected_xpath, parser_instance._NS)
    assert result == expected


def test_find_node_fqn_grouped(hdf5_groups_file):
    parser_instance = dmrparser(
        DMRPP_XML_STRINGS["hdf5_groups_file"], filepath=hdf5_groups_file
    )
    result = parser_instance.find_node_fqn("/test/group/air")
    expected = parser_instance.root.find(
        "./*[@name='test']/*[@name='group']/*[@name='air']", parser_instance._NS
    )
    assert result == expected


@pytest.mark.parametrize(
    "group_path",
    [
        ("/"),
        ("/test"),
        ("/test/group"),
    ],
)
def test_split_groups(hdf5_groups_file, group_path):
    dmrpp_instance = dmrparser(
        DMRPP_XML_STRINGS["hdf5_groups_file"], filepath=hdf5_groups_file
    )

    # get all tags in a dataset (so all tags excluding nested groups)
    dataset_tags = lambda x: [
        d for d in x if d.tag != "{" + dmrpp_instance._NS["dap"] + "}" + "Group"
    ]
    # check that contents of the split groups dataset match contents of the original dataset
    result_tags = dataset_tags(
        dmrpp_instance._split_groups(dmrpp_instance.root)[Path(group_path)]
    )
    expected_tags = dataset_tags(dmrpp_instance.find_node_fqn(group_path))
    assert result_tags == expected_tags


@pytest.mark.parametrize(
    "group,warns",
    [
        pytest.param(None, False, id="None"),
        pytest.param("/", False, id="/"),
        pytest.param("/no-such-group", True, id="/no-such-group"),
    ],
)
def test_parse_dataset(group: str | None, warns: bool, netcdf4_file):
    drmpp = dmrparser(
        DMRPP_XML_STRINGS["netcdf4_file"], filepath=f"file://{netcdf4_file}"
    )
    store = obstore_local(url=drmpp.data_filepath)

    with nullcontext() if warns else pytest.raises(BaseException, match="DID NOT WARN"):
        with pytest.warns(UserWarning, match=f"ignoring group parameter {group!r}"):
            ms = drmpp.parse_dataset(object_store=store, group=group)

    vds = ms.to_virtual_dataset()

    assert vds.sizes == {"lat": 25, "lon": 53, "time": 2920}
    assert vds.data_vars.keys() == {"air"}
    assert vds.data_vars["air"].dims == ("time", "lat", "lon")
    assert vds.attrs == {
        "Conventions": "COARDS",
        "title": "4x daily NMC reanalysis (1948)",
        "description": "Data is from NMC initialized reanalysis\n(4x/day).  These are the 0.9950 sigma level values.",
        "platform": "Model",
        "references": "http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html",
    }
    assert vds.coords.keys() == {"lat", "lon", "time"}


@pytest.mark.xfail(
    version.parse(xr.__version__) < version.parse("2025.7.1"),
    reason="Offsets in file changed",
)
def test_parse_dataset_nested(hdf5_groups_file):
    nested_groups_dmrpp = dmrparser(
        DMRPP_XML_STRINGS["hdf5_groups_file"], filepath=f"file://{hdf5_groups_file}"
    )
    store = obstore_local(url=f"file://{nested_groups_dmrpp.data_filepath}")

    vds_root_implicit = nested_groups_dmrpp.parse_dataset(
        object_store=store
    ).to_virtual_dataset(loadable_variables=[])
    vds_root = nested_groups_dmrpp.parse_dataset(
        group="/", object_store=store
    ).to_virtual_dataset(loadable_variables=[])

    xrt.assert_identical(vds_root_implicit, vds_root)
    assert vds_root.sizes == {}

    vds_g1 = nested_groups_dmrpp.parse_dataset(
        group="/test", object_store=store
    ).to_virtual_dataset(loadable_variables=[])
    assert vds_g1.sizes == {}

    vds_g2 = nested_groups_dmrpp.parse_dataset(
        group="/test/group", object_store=store
    ).to_virtual_dataset(loadable_variables=[])

    assert vds_g2.sizes == {"time": 2920, "lat": 25, "lon": 53}
    assert vds_g2.data_vars.keys() == {"air"}
    assert vds_g2.data_vars["air"].dims == ("time", "lat", "lon")


def test_parse_variable(netcdf4_file):
    parser = dmrparser(DMRPP_XML_STRINGS["netcdf4_file"], filepath=netcdf4_file)

    var = parser._parse_variable(parser.find_node_fqn("/air"))
    assert var.metadata.dtype.to_native_dtype() == "int16"
    assert var.metadata.dimension_names == ("time", "lat", "lon")
    assert var.shape == (2920, 25, 53)
    assert var.chunks == (2920, 25, 53)
    # _FillValue is encoded for array dtype
    assert var.metadata.attributes["scale_factor"] == 0.01
    assert (
        var.metadata.attributes["long_name"]
        == "4xDaily Air temperature at sigma level 995"
    )


@pytest.mark.parametrize(
    "attr_path, expected",
    [
        ("air/long_name", {"long_name": "4xDaily Air temperature at sigma level 995"}),
        ("air/scale_factor", {"scale_factor": 0.01}),
    ],
)
def test_parse_attribute(netcdf4_file, attr_path, expected):
    parser = dmrparser(DMRPP_XML_STRINGS["netcdf4_file"], filepath=netcdf4_file)

    result = parser._parse_attribute(parser.find_node_fqn(attr_path))
    assert result == expected


def test_empty_scalar_warns_container(fill_value_scalar_no_chunks_nc4_url):
    parsed_dmrpp = dmrparser(
        DMRPP_XML_STRINGS["fill_value_scalar_no_chunks_nc4_url"],
        filepath=fill_value_scalar_no_chunks_nc4_url,
    )
    store = obstore_local(url=f"file://{parsed_dmrpp.data_filepath}")
    with pytest.warns(UserWarning):
        parsed_vds = parsed_dmrpp.parse_dataset(object_store=store)
        vds_g1 = parsed_vds.to_virtual_dataset()
        assert vds_g1["data"].attrs == {"_FillValue": -999}
