import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import FITSParser
from virtualizarr.tests import requires_astropy, requires_network
from virtualizarr.tests.utils import obstore_local, obstore_s3

pytestmark = requires_astropy


def _registry(url: str) -> ObjectStoreRegistry:
    registry = ObjectStoreRegistry()
    registry.register(url, obstore_local(url=url))
    return registry


@pytest.fixture
def image_data():
    return np.arange(6 * 5, dtype=">i2").reshape(6, 5)


@pytest.fixture
def cube_data():
    return np.arange(2 * 3 * 4, dtype=">f4").reshape(2, 3, 4)


@pytest.fixture
def fits_file(tmp_path, image_data, cube_data):
    """A FITS file exercising each kind of HDU the parser handles."""
    from astropy.io import fits

    scaled = fits.ImageHDU(data=np.arange(12, dtype=">i2").reshape(3, 4), name="SCALED")
    scaled.header["BSCALE"] = 2.0
    scaled.header["BZERO"] = 10.0

    ascii_table = fits.TableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="x", format="E12.4", array=np.array([1.5, 2.5])),
                fits.Column(name="n", format="I5", array=np.array([7, 8])),
            ]
        ),
        name="ATAB",
    )

    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "TEST"

    path = tmp_path / "test.fits"
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=image_data, name="SCI"),
            scaled,
            fits.ImageHDU(data=cube_data, name="CUBE"),
            ascii_table,
        ]
    ).writeto(path)
    return path


@pytest.fixture
def fits_store(fits_file):
    url = fits_file.as_uri()
    return url, _registry(url)


def test_all_data_hdus_become_arrays(fits_store):
    url, registry = fits_store
    with open_virtual_dataset(url=url, registry=registry, parser=FITSParser()) as vds:
        assert list(vds.variables) == ["SCI", "SCALED", "CUBE", "ATAB"]


def test_primary_header_becomes_group_attributes(fits_store):
    """The primary HDU carries no data here, so its header describes the file."""
    url, registry = fits_store
    with open_virtual_dataset(url=url, registry=registry, parser=FITSParser()) as vds:
        assert vds.attrs["TELESCOP"] == "TEST"


def test_hdu_header_becomes_array_attributes(fits_store):
    url, registry = fits_store
    with open_virtual_dataset(url=url, registry=registry, parser=FITSParser()) as vds:
        assert vds["SCI"].attrs["EXTNAME"] == "SCI"


def test_axes_are_named_per_hdu(fits_store):
    """HDUs disagree about axis lengths, so each HDU's axes are named separately."""
    url, registry = fits_store
    with open_virtual_dataset(url=url, registry=registry, parser=FITSParser()) as vds:
        assert vds["SCI"].dims == ("SCI_y", "SCI_x")
        assert vds["CUBE"].dims == ("CUBE_z", "CUBE_y", "CUBE_x")
        assert vds["ATAB"].dims == ("ATAB_row",)


def test_skip_variables(fits_store):
    url, registry = fits_store
    parser = FITSParser(skip_variables=["SCALED", "ATAB"])
    with open_virtual_dataset(url=url, registry=registry, parser=parser) as vds:
        assert list(vds.variables) == ["SCI", "CUBE"]


def test_group_must_be_root(fits_store):
    url, registry = fits_store
    with pytest.raises(ValueError, match="only a root group"):
        FITSParser(group="nested")(url, registry)


@pytest.mark.parametrize("name", ["SCI", "CUBE"])
def test_image_roundtrips(fits_store, image_data, cube_data, name):
    url, registry = fits_store
    expected = {"SCI": image_data, "CUBE": cube_data}[name]
    manifest_store = FITSParser()(url, registry)
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        np.testing.assert_array_equal(ds[name].values, expected)
        # FITS stores big-endian, but decoding it yields the equivalent native dtype.
        assert ds[name].dtype == expected.dtype.newbyteorder("=")


def test_scaled_image_applies_bscale_and_bzero(fits_store):
    """FITS decodes stored integers as BZERO + BSCALE * stored."""
    url, registry = fits_store
    manifest_store = FITSParser()(url, registry)
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        expected = np.arange(12).reshape(3, 4) * 2.0 + 10.0
        np.testing.assert_allclose(ds["SCALED"].values, expected)


def test_ascii_table_columns_are_decoded(fits_store):
    url, registry = fits_store
    manifest_store = FITSParser()(url, registry)
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        table = ds["ATAB"].values
        np.testing.assert_allclose(table["x"], [1.5, 2.5])
        np.testing.assert_array_equal(table["n"], [7, 8])


def test_blank_becomes_fill_value(tmp_path):
    """BLANK marks undefined pixels of an integer image, so xarray should mask them."""
    from astropy.io import fits

    data = np.array([[1, -999], [3, 4]], dtype=">i2")
    hdu = fits.ImageHDU(data=data, name="SCI")
    hdu.header["BLANK"] = -999
    path = tmp_path / "blank.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    url = path.as_uri()
    with open_virtual_dataset(
        url=url, registry=_registry(url), parser=FITSParser()
    ) as vds:
        assert vds["SCI"].attrs["_FillValue"] == -999

    manifest_store = FITSParser()(url, _registry(url))
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        assert np.isnan(ds["SCI"].values[0, 1])
        np.testing.assert_array_equal(ds["SCI"].values[1], [3.0, 4.0])


def test_blank_is_scaled_alongside_the_data(tmp_path):
    """BLANK is a stored value, so it scales with the array it labels."""
    from astropy.io import fits

    hdu = fits.ImageHDU(data=np.array([[1, -99]], dtype=">i2"), name="SCI")
    hdu.header["BLANK"] = -99
    hdu.header["BSCALE"] = 2.0
    hdu.header["BZERO"] = 10.0
    path = tmp_path / "blank_scaled.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    url = path.as_uri()
    manifest_store = FITSParser()(url, _registry(url))
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        # A float _FillValue reaches the attribute base64-encoded, as xarray's zarr
        # backend expects, so read it back off the decoded variable.
        assert ds["SCI"].encoding["_FillValue"] == 10.0 + 2.0 * -99
        np.testing.assert_array_equal(ds["SCI"].values[0, 0], 12.0)
        assert np.isnan(ds["SCI"].values[0, 1])


def test_binary_table_raises(tmp_path):
    """Zarr v3 cannot record that a table's columns are stored big-endian."""
    from astropy.io import fits

    path = tmp_path / "bintable.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                fits.ColDefs(
                    [
                        fits.Column(
                            name="flux",
                            format="E",
                            array=np.array([1.5, 2.5], dtype=">f4"),
                        )
                    ]
                ),
                name="SPEC",
            ),
        ]
    ).writeto(path)

    url = path.as_uri()
    registry = _registry(url)

    with pytest.raises(ValueError, match="byte-swapped"):
        FITSParser()(url, registry)

    parser = FITSParser(skip_variables=["SPEC"])
    assert list(parser(url, registry)._group._members) == []


def test_repeated_extension_names_are_disambiguated(tmp_path, image_data):
    """FITS allows several HDUs to share an EXTNAME, but Zarr array names must differ."""
    from astropy.io import fits

    path = tmp_path / "repeated.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(data=image_data, name="SCI"),
            fits.ImageHDU(data=image_data, name="SCI"),
        ]
    ).writeto(path)

    url = path.as_uri()
    with open_virtual_dataset(
        url=url, registry=_registry(url), parser=FITSParser()
    ) as vds:
        assert list(vds.variables) == ["SCI", "SCI_2"]


@requires_network
def test_open_hubble_data():
    # data from https://registry.opendata.aws/hst/
    url = "s3://stpubdata/hst/public/f05i/f05i0201m/f05i0201m_a1f.fits"
    store = obstore_s3(url=url, region="us-east-1")
    registry = ObjectStoreRegistry()
    registry.register(url, store)
    with open_virtual_dataset(
        url=url,
        registry=registry,
        parser=FITSParser(),
    ) as vds:
        assert isinstance(vds, Dataset)
        # The file pairs the image with an ASCII table of its WCS keywords.
        assert list(vds.variables) == ["PRIMARY", "f05i0201m.a1h.tab"]
        var = vds["PRIMARY"].variable
        assert var.sizes == {"PRIMARY_y": 17, "PRIMARY_x": 589}
        assert var.dtype == "int32"


@requires_network
def test_hubble_ascii_table_is_decoded():
    """The table's columns do not tile the row, so TBCOL is what places them."""
    url = "s3://stpubdata/hst/public/f05i/f05i0201m/f05i0201m_a1f.fits"
    registry = ObjectStoreRegistry()
    registry.register(url, obstore_s3(url=url, region="us-east-1"))

    manifest_store = FITSParser()(url, registry)
    with xr.open_zarr(manifest_store, zarr_format=3, consolidated=False) as ds:
        table = ds["f05i0201m.a1h.tab"].values
        assert table.dtype.names == (
            "DATAMIN",
            "DATAMAX",
            "CRPIX1",
            "CD1_1",
            "CRVAL1",
            "CTYPE1",
        )
        # Written with a Fortran "D" exponent, which needs translating to parse.
        np.testing.assert_allclose(table["CRVAL1"][0], 168717651.634)
        assert table["CTYPE1"][0].strip() == b"SECONDS"
