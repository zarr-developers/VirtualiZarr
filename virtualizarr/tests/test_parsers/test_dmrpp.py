from xml.etree import ElementTree as ET

import pytest
import requests
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.parsers import DMRPPParser, HDFParser
from virtualizarr.tests import requires_network, requires_pydap, slow_test
from virtualizarr.tests.utils import obstore_local, obstore_s3
from virtualizarr.xarray import open_virtual_dataset

pytest.importorskip("pydap")
from pydap.virtualizarr.parser import DMRParser  # noqa: E402

urls = [
    (
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc",
        "s3://its-live-data/test-space/cloud-experiments/dmrpp/20240826090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc.dmrpp",
    )
    # TODO: later add MUR, SWOT, TEMPO and others by using kerchunk JSON to read refs (rather than reading the whole netcdf file)
]


@requires_pydap
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


<<<<<<< HEAD
@pytest.mark.xfail(
    reason="See https://github.com/zarr-developers/VirtualiZarr/issues/904.",
)
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


<<<<<<< HEAD
# def test_parse_variable(netcdf4_file):
#     parser = dmrparser(DMRPP_XML_STRINGS["netcdf4_file"], filepath=netcdf4_file)

    # var = parser._parse_variable(parser.find_node_fqn("/air"))
    # assert var.metadata.dtype.to_native_dtype() == "int16"
    # assert var.metadata.dimension_names == ("time", "lat", "lon")
    # assert var.shape == (2920, 25, 53)
    # assert var.metadata.chunks == (2920, 25, 53)
    # # _FillValue is encoded for array dtype
    # assert var.metadata.attributes["scale_factor"] == 0.01
    # assert (
    #     var.metadata.attributes["long_name"]
    #     == "4xDaily Air temperature at sigma level 995"
    # )


# @pytest.mark.parametrize(
#     "attr_path, expected",
#     [
#         ("air/long_name", {"long_name": "4xDaily Air temperature at sigma level 995"}),
#         ("air/scale_factor", {"scale_factor": 0.01}),
#     ],
# )
# def test_parse_attribute(netcdf4_file, attr_path, expected):
#     parser = dmrparser(DMRPP_XML_STRINGS["netcdf4_file"], filepath=netcdf4_file)

#     result = parser._parse_attribute(parser.find_node_fqn(attr_path))
#     assert result == expected


=======
>>>>>>> 79b9a79 (rebase)
=======
@requires_pydap
>>>>>>> 5fa3d8f (update dmrpp migration - soft imports and code refactor)
def test_dmrpp_simple(dmrpp_xml_simple):
    """Test parsing a simple valid DMR++ XML creates virtual chunk manifests."""
    parser = dmrparser(dmrpp_xml_simple, filepath="file:///simple.nc")

    # Parse dataset
    manifest_store = parser.parse_dataset(
        object_store=obstore_local(url="file:///"), group="/"
    )

    # Verify manifest store is created
    assert manifest_store is not None

    # Verify dimensions are parsed correctly from the manifest group (accessing private _group for testing)
    manifest_group = manifest_store._group
    assert manifest_group is not None

    # The manifest group should have arrays with correct dimensions
    assert "temperature" in manifest_group.arrays
    temperature_array = manifest_group.arrays["temperature"]
    assert temperature_array.shape == (25, 53)

    # Verify chunk manifest exists
    temperature_manifest = temperature_array.manifest.dict()
    assert len(temperature_manifest) > 0

    # Verify chunk manifest entries have expected structure
    for key, chunk_info in temperature_manifest.items():
        assert "path" in chunk_info
        assert "offset" in chunk_info
        assert "length" in chunk_info
        assert chunk_info["path"] == "file:///simple.nc"
        assert isinstance(chunk_info["offset"], int)
        assert isinstance(chunk_info["length"], int)


@requires_pydap
def test_dmrpp_missing_attrib_validation(dmrpp_xml_with_missing_attrib):
    """Test that validation issues are accumulated for missing attributes."""
    parser = dmrparser(
        dmrpp_xml_with_missing_attrib, filepath="file:///validation_test.nc"
    )

    # Parse dataset - this should accumulate validation issues
    manifest_store = parser.parse_dataset(
        object_store=obstore_local(url="file:///"), group="/"
    )

    # Verify that validation issues were accumulated
    assert len(parser._validation_issues) > 0

    # Check that the issues mention missing attributes
    assert any(
        "Missing required attribute 'name'" in issue
        for issue in parser._validation_issues
    )

    # Verify manifest store was still created (parser continues despite issues)
    assert manifest_store is not None
    assert manifest_store._group is not None


@requires_network
@requires_pydap
def test_inlinevalue():
    """
    Test that inline values can be parsed into manifest
    """
    expected_bytes = b"AAAAAAAAAAABAAAAAAAAAAIAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAAUAAAAAAAAABgAAAAAAAAAHAAAAAAAAAAgAAAAAAAAACQAAAAAAAAA="

    dmrpp_file = (
        "http://test.opendap.org/opendap/data/dmrpp/compact_lowlevel.h5.dmrpp.file"
    )
    session = requests.Session()
    dmrpp = session.get(dmrpp_file).content.decode()
    parser = dmrparser(dmrpp, filepath="file:///")
    store = obstore_local(url=parser.data_filepath)
    ms = parser.parse_dataset(object_store=store)
    vds = ms.to_virtual_dataset()
    assert "my_dataset" in vds
    assert ms._group["my_dataset"]._manifest._inlined == {(0, 0): expected_bytes}
