import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import NetCDF3Parser
from virtualizarr.parsers.netcdf3 import _parse_header
from virtualizarr.tests import requires_network
from virtualizarr.tests.utils import obstore_http


def _open_virtual(path, registry):
    parser = NetCDF3Parser()
    return parser(url=f"file://{path}", registry=registry)


def _assert_matches_xarray(path, registry, **open_kwargs):
    """The virtual dataset must read back exactly what xarray reads natively."""
    with (
        _open_virtual(path, registry) as manifest_store,
        xr.open_dataset(path, **open_kwargs) as expected,
    ):
        observed = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        xrt.assert_identical(observed.load(), expected.load())


def test_read_netcdf3(netcdf3_file, local_registry):
    filepath = netcdf3_file()
    with (
        _open_virtual(filepath, local_registry) as manifest_store,
        xr.open_dataset(filepath) as expected,
    ):
        observed = xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
        assert isinstance(observed, xr.Dataset)
        assert list(observed.variables.keys()) == ["foo"]
        xrt.assert_identical(observed.load(), expected.load())


class TestCDFVersions:
    """Every netCDF classic format: CDF-1, CDF-2 (64-bit offset), CDF-5 (64-bit data)."""

    def test_matches_xarray(self, netcdf3_variant_file, local_registry):
        path, _ = netcdf3_variant_file
        _assert_matches_xarray(path, local_registry)

    def test_version_detected(self, netcdf3_variant_file):
        path, cdf_version = netcdf3_variant_file
        with open(path, "rb") as f:
            header = _parse_header(f)
        assert header.version == cdf_version

    def test_record_variables_are_chunked_per_record(
        self, netcdf3_variant_file, local_registry
    ):
        """Record variables are interleaved on disk, so each record is its own chunk."""
        path, _ = netcdf3_variant_file
        with _open_virtual(path, local_registry) as store:
            rec = store._group["rec_f8"]
            assert rec.shape == (3, 4)
            assert rec.metadata.chunks == (1, 4)
            assert rec.manifest.shape_chunk_grid == (3, 1)

    def test_contiguous_variables_are_a_single_chunk(
        self, netcdf3_variant_file, local_registry
    ):
        path, _ = netcdf3_variant_file
        with _open_virtual(path, local_registry) as store:
            arr = store._group["contiguous"]
            assert arr.shape == (4, 3)
            assert arr.metadata.chunks == (4, 3)
            assert arr.manifest.shape_chunk_grid == (1, 1)

    def test_record_chunks_are_strided_by_record_size(
        self, netcdf3_variant_file, local_registry
    ):
        """Consecutive records of one variable sit exactly one record size apart."""
        path, _ = netcdf3_variant_file
        with open(path, "rb") as f:
            header = _parse_header(f)
        with _open_virtual(path, local_registry) as store:
            entries = store._group["rec_f8"].manifest.dict()
            offsets = [entries[f"{i}.0"]["offset"] for i in range(3)]
            assert np.all(np.diff(offsets) == header.recsize)

    def test_scalar_variable(self, netcdf3_variant_file, local_registry):
        path, _ = netcdf3_variant_file
        with _open_virtual(path, local_registry) as store:
            observed = xr.open_dataset(
                store, engine="zarr", consolidated=False, zarr_format=3
            )
            assert observed["scalar"].shape == ()
            assert observed["scalar"].load().item() == 42

    def test_attributes(self, netcdf3_variant_file, local_registry):
        path, _ = netcdf3_variant_file
        with _open_virtual(path, local_registry) as store:
            observed = xr.open_dataset(
                store, engine="zarr", consolidated=False, zarr_format=3
            )
            assert observed.attrs["title"] == "netCDF3 test file"
            assert observed.attrs["answer"] == 42
            assert observed["contiguous"].attrs["units"] == "kelvin"

    def test_skip_variables(self, netcdf3_variant_file, local_registry):
        path, _ = netcdf3_variant_file
        parser = NetCDF3Parser(skip_variables=["contiguous", "rec_f8"])
        with parser(url=f"file://{path}", registry=local_registry) as store:
            assert "contiguous" not in store._group
            assert "rec_f8" not in store._group
            assert "rec_i2" in store._group


class TestCDF5Types:
    """CDF-5 adds unsigned and 64-bit integer types the earlier formats lack."""

    @pytest.mark.parametrize(
        "name, dtype, values",
        [
            ("big_uint", ">u8", [1, 2, 2**40]),
            ("big_int", ">i8", [-(2**40), 0, 2**40]),
            ("ushort", ">u2", [0, 1, 65535]),
        ],
    )
    def test_wide_integer_types(
        self, netcdf3_variant_file, local_registry, name, dtype, values
    ):
        path, cdf_version = netcdf3_variant_file
        if cdf_version != 5:
            pytest.skip("64-bit and unsigned types only exist in CDF-5")
        with _open_virtual(path, local_registry) as store:
            arr = store._group[name]
            assert arr.dtype == np.dtype(dtype)
            observed = xr.open_dataset(
                store, engine="zarr", consolidated=False, zarr_format=3
            )
            np.testing.assert_array_equal(observed[name].load().values, values)

    def test_cdf5_only_types_rejected_in_cdf1(self, tmp_path):
        """A CDF-1 file claiming a CDF-5-only type is malformed, not silently read."""
        path = tmp_path / "bad_type.nc"
        # NC_UBYTE (7) as the type of a variable in a CDF-1 file.
        header = bytearray(b"CDF\x01")
        header += (0).to_bytes(4, "big")  # numrecs
        header += (10).to_bytes(4, "big") + (1).to_bytes(4, "big")  # NC_DIMENSION, 1
        header += (1).to_bytes(4, "big") + b"x\x00\x00\x00"  # dim name "x"
        header += (3).to_bytes(4, "big")  # dim length
        header += (0).to_bytes(4, "big") * 2  # no global attributes
        header += (11).to_bytes(4, "big") + (1).to_bytes(4, "big")  # NC_VARIABLE, 1
        header += (1).to_bytes(4, "big") + b"v\x00\x00\x00"  # var name "v"
        header += (1).to_bytes(4, "big") + (0).to_bytes(4, "big")  # 1 dim, dimid 0
        header += (0).to_bytes(4, "big") * 2  # no variable attributes
        header += (7).to_bytes(4, "big")  # nc_type NC_UBYTE -- CDF-5 only
        header += (3).to_bytes(4, "big") + (100).to_bytes(4, "big")  # vsize, begin
        path.write_bytes(bytes(header) + b"\x00" * 100)

        with open(path, "rb") as f:
            with pytest.raises(ValueError, match="only valid in CDF-5"):
                _parse_header(f)


class TestFillValue:
    def test_fill_value_is_preserved(self, netcdf3_file, local_registry):
        """Sentinel values must decode to NaN, as they do reading the file natively.

        The kerchunk-backed parser dropped `_FillValue` from the array metadata, so
        sentinels survived into the loaded data instead of being masked. Closes
        [#982](https://github.com/zarr-developers/VirtualiZarr/issues/982).
        """
        ds = xr.Dataset({"t": ("x", [1.0, np.nan, 3.0, np.nan])})
        ds["t"].encoding["_FillValue"] = -999.0
        path = netcdf3_file(ds, name="fill_value.nc")

        with _open_virtual(path, local_registry) as store:
            observed = xr.open_dataset(
                store, engine="zarr", consolidated=False, zarr_format=3
            ).load()
        np.testing.assert_array_equal(observed["t"].values, [1.0, np.nan, 3.0, np.nan])
        assert observed["t"].encoding["_FillValue"] == -999.0


class TestRecordPadding:
    def test_single_record_variable_is_unpadded(
        self, netcdf3_single_record_var_file, local_registry
    ):
        """With one record variable its 3-byte record slice gets no padding."""
        path = netcdf3_single_record_var_file
        with open(path, "rb") as f:
            header = _parse_header(f)
        assert header.recsize == 3
        _assert_matches_xarray(path, local_registry)

    def test_multiple_record_variables_are_padded(
        self, netcdf3_variant_file, local_registry
    ):
        """With several record variables each slice is padded to a multiple of 4."""
        path, _ = netcdf3_variant_file
        with open(path, "rb") as f:
            header = _parse_header(f)
        # rec_f8 is 4*8 bytes, rec_i2 is 2 bytes padded to 4, rec_i1 is 3 padded to 4.
        assert header.recsize == 32 + 4 + 4


class TestStreamingNumrecs:
    def test_numrecs_recovered_from_file_size(
        self, netcdf3_single_record_var_file, local_registry
    ):
        """A writer that never finalized numrecs leaves it as all 1-bits."""
        path = netcdf3_single_record_var_file
        raw = bytearray(path.read_bytes())
        assert int.from_bytes(raw[4:8], "big") == 4
        raw[4:8] = (0xFFFFFFFF).to_bytes(4, "big")
        path.write_bytes(bytes(raw))

        with _open_virtual(path, local_registry) as store:
            observed = xr.open_dataset(
                store, engine="zarr", consolidated=False, zarr_format=3
            )
            assert observed["only"].shape == (4, 3)
            np.testing.assert_array_equal(
                observed["only"].load().values, np.arange(12).reshape(4, 3)
            )


class TestMalformedFiles:
    def test_not_a_netcdf3_file(self, tmp_path, local_registry):
        path = tmp_path / "nope.nc"
        path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 100)
        with pytest.raises(ValueError, match="not a netCDF3 file"):
            _open_virtual(path, local_registry)

    def test_unknown_cdf_version(self, tmp_path, local_registry):
        path = tmp_path / "future.nc"
        path.write_bytes(b"CDF\x09" + b"\x00" * 100)
        with pytest.raises(ValueError, match="unknown netCDF classic format version"):
            _open_virtual(path, local_registry)

    def test_truncated_header(self, tmp_path, local_registry):
        path = tmp_path / "truncated.nc"
        path.write_bytes(b"CDF\x01" + b"\x00" * 4 + (10).to_bytes(4, "big"))
        with pytest.raises(ValueError, match="truncated"):
            _open_virtual(path, local_registry)

    def test_subgroup_rejected(self, netcdf3_file, local_registry):
        """A netCDF3 file is flat, so there is no subgroup to open."""
        parser = NetCDF3Parser(group="subgroup")
        with pytest.raises(ValueError, match="only a root group"):
            parser(url=f"file://{netcdf3_file()}", registry=local_registry)


@requires_network
def test_read_http_netcdf3():
    url = "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc"
    store = obstore_http(url=url)
    registry = ObjectStoreRegistry({url: store})
    parser = NetCDF3Parser()
    with open_virtual_dataset(url=url, parser=parser, registry=registry) as vds:
        assert isinstance(vds, xr.Dataset)
        assert set(vds.dims) == {"lat", "lon", "time"}
