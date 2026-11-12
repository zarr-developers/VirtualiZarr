import pytest
import requests
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.manifests import ManifestStore
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


@slow_test
@requires_network
@requires_pydap
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
@requires_pydap
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


def local_registry() -> ObjectStoreRegistry:
    return ObjectStoreRegistry({"file://": LocalStore()})


@requires_pydap
def test_dmrpp_parser_local(dmrpp_simple_url):
    """DMRPPParser reads a local DMR++ file through the registry."""
    manifest_store = DMRPPParser()(url=dmrpp_simple_url, registry=local_registry())

    assert isinstance(manifest_store, ManifestStore)
    temperature = manifest_store._group.arrays["temperature"]
    assert temperature.shape == (25, 53)
    assert temperature.metadata.attributes["units"] == "K"

    # Chunk references point at the data file next to the .dmrpp file.
    expected_path = dmrpp_simple_url.removesuffix(".dmrpp")
    assert temperature.manifest.dict() == {
        "0.0": {"path": expected_path, "offset": 100, "length": 5300}
    }


@requires_pydap
def test_dmrpp_parser_skip_variables(dmrpp_grouped_url):
    """The skip_variables argument is passed through to pydap."""
    registry = local_registry()

    manifest_store = DMRPPParser()(url=dmrpp_grouped_url, registry=registry)
    assert "x" in manifest_store._group.arrays

    manifest_store = DMRPPParser(skip_variables=["x"])(
        url=dmrpp_grouped_url, registry=registry
    )
    assert "x" not in manifest_store._group.arrays


@requires_pydap
@pytest.mark.parametrize(
    "group, expected_vars",
    [
        (None, {"x"}),
        ("/", {"x"}),
        ("obs", {"pressure"}),
        ("/obs", {"pressure"}),
    ],
)
def test_dmrpp_parser_group(dmrpp_grouped_url, group, expected_vars):
    """The group argument selects which DMR++ group becomes the ManifestStore root."""
    manifest_store = DMRPPParser(group=group)(
        url=dmrpp_grouped_url, registry=local_registry()
    )
    assert set(manifest_store._group.arrays) == expected_vars


@requires_pydap
def test_dmrpp_open_virtual_dataset_local(dmrpp_grouped_url):
    """DMRPPParser works through the public open_virtual_dataset API."""
    with open_virtual_dataset(
        url=dmrpp_grouped_url,
        registry=local_registry(),
        parser=DMRPPParser(group="obs"),
        loadable_variables=[],
    ) as vds:
        assert set(vds.data_vars) | set(vds.coords) == {"pressure"}
        assert vds["pressure"].shape == (3,)
        assert vds["pressure"].attrs["units"] == "hPa"


@requires_pydap
def test_dmrpp_simple(dmrpp_xml_simple):
    """Test parsing a simple valid DMR++ XML creates virtual chunk manifests."""
    parser = DMRParser(
        dmrpp_xml_simple, object_store=obstore_local(url="file:///simple.nc")
    )

    # Parse dataset
    manifest_store = parser.parse_dataset(group="/")

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
    parser = DMRParser(
        dmrpp_xml_with_missing_attrib,
        object_store=obstore_local(url="file:///validation_test.nc"),
    )

    # Parse dataset - this should accumulate validation issues
    manifest_store = parser.parse_dataset(group="/")

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
    parser = DMRParser(
        dmrpp, object_store=obstore_local(url="file:///compact_lowlevel.h5")
    )
    ms = parser.parse_dataset()
    vds = ms.to_virtual_dataset()
    assert "my_dataset" in vds
    assert ms._group["my_dataset"]._manifest._inlined == {(0, 0): expected_bytes}
