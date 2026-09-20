import gzip
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from xarray import Dataset

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import ENVIParser
from virtualizarr.parsers.envi import _parse_envi_header

DATA = Path(__file__).resolve().parents[1] / "data" / "envi"

# Real ENVI test fixtures vendored from GDAL's autotest suite (public domain / freely
# redistributable), covering the three interleave schemes, both byte orders, and
# optional whole-file gzip compression:
# https://github.com/OSGeo/gdal/tree/master/autotest/gdrivers/data/envi
BSQ = DATA / "envi_rgbsmall_bsq.img"
BIL = DATA / "envi_rgbsmall_bil.img"
BIP = DATA / "envi_rgbsmall_bip.img"
AEA = DATA / "aea.dat"
AEA_COMPRESSED = DATA / "aea_compressed.dat"


def _registry_and_url(fixture: Path) -> tuple[ObjectStoreRegistry, str]:
    registry = ObjectStoreRegistry({"file://": LocalStore()})
    return registry, fixture.as_uri()


def _read_reference_array(hdr_path: Path, dat_path: Path) -> np.ndarray:
    """Read an ENVI file directly with numpy, independent of ENVIParser, as ground truth."""

    fields = _parse_envi_header(hdr_path.read_text())
    samples = int(fields["samples"])
    lines = int(fields["lines"])
    bands = int(fields["bands"])
    byte_order = int(fields.get("byte order", "0"))
    header_offset = int(fields.get("header offset", "0"))
    interleave = fields["interleave"].strip().lower()

    dtype_map = {
        1: "u1",
        2: "i2",
        3: "i4",
        4: "f4",
        5: "f8",
        12: "u2",
        13: "u4",
        14: "i8",
        15: "u8",
    }
    endian = ">" if byte_order == 1 else "<"
    dtype = np.dtype(endian + dtype_map[int(fields["data type"])])

    raw = dat_path.read_bytes()
    if int(fields.get("file compression", "0")) == 1:
        raw = gzip.decompress(raw[header_offset:])
        header_offset = 0

    arr = np.frombuffer(raw, dtype=dtype, offset=header_offset)
    shapes = {
        "bsq": (bands, lines, samples),
        "bil": (lines, bands, samples),
        "bip": (lines, samples, bands),
    }
    return arr.reshape(shapes[interleave])


@pytest.mark.parametrize("fixture", [BSQ, BIL, BIP])
def test_envi_interleave_schemes_match_reference(fixture: Path) -> None:
    registry, url = _registry_and_url(fixture)
    parser = ENVIParser()
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, Dataset)
        [var_name] = list(vds.data_vars)
        manifest_store = parser(url, registry)
        with xr.open_dataset(
            manifest_store, engine="zarr", zarr_format=3, consolidated=False
        ) as actual:
            expected = _read_reference_array(fixture.with_suffix(".hdr"), fixture)
            np.testing.assert_array_equal(actual[var_name].to_numpy(), expected)


def test_envi_big_endian() -> None:
    registry, url = _registry_and_url(AEA)
    parser = ENVIParser()
    manifest_store = parser(url, registry)
    with xr.open_dataset(
        manifest_store, engine="zarr", zarr_format=3, consolidated=False
    ) as actual:
        [var_name] = list(actual.data_vars)
        expected = _read_reference_array(AEA.with_suffix(".hdr"), AEA)
        np.testing.assert_array_equal(actual[var_name].to_numpy(), expected)
        assert actual[var_name].attrs["sensor_type"] == "Landsat TM"


def test_envi_gzip_compressed() -> None:
    registry, url = _registry_and_url(AEA_COMPRESSED)
    parser = ENVIParser()
    manifest_store = parser(url, registry)
    with xr.open_dataset(
        manifest_store, engine="zarr", zarr_format=3, consolidated=False
    ) as actual:
        [var_name] = list(actual.data_vars)
        expected = _read_reference_array(
            AEA_COMPRESSED.with_suffix(".hdr"), AEA_COMPRESSED
        )
        np.testing.assert_array_equal(actual[var_name].to_numpy(), expected)


def test_envi_default_chunking_splits_along_outer_axis() -> None:
    registry, url = _registry_and_url(BSQ)
    parser = ENVIParser()
    manifest_store = parser(url, registry)
    vds = manifest_store.to_virtual_dataset()
    [var_name] = list(vds.data_vars)
    manifest = vds[var_name].data.manifest
    # BSQ: shape is (band, y, x) = (3, 49, 50), one chunk per band expected
    assert manifest.shape_chunk_grid == (3, 1, 1)


def test_envi_split_chunks_false_gives_single_chunk() -> None:
    registry, url = _registry_and_url(BSQ)
    parser = ENVIParser(split_chunks_along_outer_axis=False)
    manifest_store = parser(url, registry)
    vds = manifest_store.to_virtual_dataset()
    [var_name] = list(vds.data_vars)
    manifest = vds[var_name].data.manifest
    assert manifest.shape_chunk_grid == (1, 1, 1)


@pytest.mark.parametrize("split_chunks_along_outer_axis", [True, False])
def test_envi_chunking_reads_match_reference(
    split_chunks_along_outer_axis: bool,
) -> None:
    registry, url = _registry_and_url(BSQ)
    parser = ENVIParser(split_chunks_along_outer_axis=split_chunks_along_outer_axis)
    manifest_store = parser(url, registry)
    with xr.open_dataset(
        manifest_store, engine="zarr", zarr_format=3, consolidated=False
    ) as actual:
        [var_name] = list(actual.data_vars)
        expected = _read_reference_array(BSQ.with_suffix(".hdr"), BSQ)
        np.testing.assert_array_equal(actual[var_name].to_numpy(), expected)


def test_envi_gzip_forces_single_chunk_even_when_split_requested() -> None:
    registry, url = _registry_and_url(AEA_COMPRESSED)
    parser = ENVIParser(split_chunks_along_outer_axis=True)
    manifest_store = parser(url, registry)
    vds = manifest_store.to_virtual_dataset()
    [var_name] = list(vds.data_vars)
    manifest = vds[var_name].data.manifest
    assert manifest.shape_chunk_grid == (1, 1, 1)


def test_envi_dimension_names_follow_interleave() -> None:
    registry, url = _registry_and_url(BIL)
    parser = ENVIParser()
    manifest_store = parser(url, registry)
    vds = manifest_store.to_virtual_dataset()
    [var_name] = list(vds.data_vars)
    assert vds[var_name].dims == ("y", "band", "x")


def test_envi_missing_required_field(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text("ENVI\nsamples = 4\nlines = 4\ninterleave = bsq\ndata type = 1\n")
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(ValueError, match="bands"):
        parser(dat.as_uri(), registry)


def test_envi_unsupported_interleave(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 1\ndata type = 1\ninterleave = bip2\n"
    )
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(ValueError, match="interleave"):
        parser(dat.as_uri(), registry)


def test_envi_unsupported_data_type(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 1\ndata type = 99\ninterleave = bsq\n"
    )
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(ValueError, match="data type"):
        parser(dat.as_uri(), registry)


def test_envi_unsupported_file_type(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 1\ndata type = 1\ninterleave = bsq\n"
        "file type = ENVI Spectral Library\n"
    )
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(NotImplementedError, match="file type"):
        parser(dat.as_uri(), registry)


def test_envi_frame_offsets_unsupported(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 1\ndata type = 1\ninterleave = bsq\n"
        "major frame offsets = {2, 2}\n"
    )
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(NotImplementedError, match="frame"):
        parser(dat.as_uri(), registry)


def test_envi_size_mismatch(tmp_path) -> None:
    hdr = tmp_path / "broken.hdr"
    hdr.write_text(
        "ENVI\nsamples = 4\nlines = 4\nbands = 1\ndata type = 1\ninterleave = bsq\n"
    )
    dat = tmp_path / "broken.dat"
    dat.write_bytes(b"\x00" * 8)  # should be 16 bytes

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(ValueError, match="does not match"):
        parser(dat.as_uri(), registry)


def test_envi_missing_header(tmp_path) -> None:
    dat = tmp_path / "nohdr.dat"
    dat.write_bytes(b"\x00" * 16)

    registry = ObjectStoreRegistry({"file://": LocalStore()})
    parser = ENVIParser()
    with pytest.raises(FileNotFoundError, match="header"):
        parser(dat.as_uri(), registry)
