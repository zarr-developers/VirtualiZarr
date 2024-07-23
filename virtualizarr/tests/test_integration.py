import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray


@pytest.mark.parametrize(
    "inline_threshold, vars_to_inline",
    [
        (5e2, ["lat", "lon"]),
        (5e4, ["lat", "lon", "time"]),
        pytest.param(
            5e7,
            ["lat", "lon", "time", "air"],
            marks=pytest.mark.xfail(reason="scale factor encoding"),
        ),
    ],
)
def test_numpy_arrays_to_inlined_kerchunk_refs(
    netcdf4_file, inline_threshold, vars_to_inline
):
    from kerchunk.hdf import SingleHdf5ToZarr

    # inline_threshold is chosen to test inlining only the variables listed in vars_to_inline
    expected = SingleHdf5ToZarr(
        netcdf4_file, inline_threshold=int(inline_threshold)
    ).translate()

    # loading the variables should produce same result as inlining them using kerchunk
    vds = open_virtual_dataset(
        netcdf4_file, loadable_variables=vars_to_inline, indexes={}
    )
    refs = vds.virtualize.to_kerchunk(format="dict")

    # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/TomNicholas/VirtualiZarr/pull/73#issuecomment-2040931202
    # assert refs == expected
    assert refs["refs"]["air/0.0.0"] == expected["refs"]["air/0.0.0"]
    assert refs["refs"]["lon/0"] == expected["refs"]["lon/0"]
    assert refs["refs"]["lat/0"] == expected["refs"]["lat/0"]
    assert refs["refs"]["time/0"] == expected["refs"]["time/0"]


@pytest.mark.parametrize("format", ["dict", "json", "parquet"])
class TestKerchunkRoundtrip:
    def test_kerchunk_roundtrip_no_concat(self, tmpdir, format):
        # set up example xarray dataset
        ds = xr.tutorial.open_dataset("air_temperature", decode_times=False)

        # save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(f"{tmpdir}/air.nc")

        # use open_dataset_via_kerchunk to read it as references
        vds = open_virtual_dataset(f"{tmpdir}/air.nc", indexes={})

        if format == "dict":
            # write those references to an in-memory kerchunk-formatted references dictionary
            ds_refs = vds.virtualize.to_kerchunk(format=format)

            # use fsspec to read the dataset from the kerchunk references dict
            roundtrip = xr.open_dataset(ds_refs, engine="kerchunk", decode_times=False)
        else:
            # write those references to disk as kerchunk references format
            vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

            # use fsspec to read the dataset from disk via the kerchunk references
            roundtrip = xr.open_dataset(
                f"{tmpdir}/refs.{format}", engine="kerchunk", decode_times=False
            )

        # assert identical to original dataset
        xrt.assert_identical(roundtrip, ds)

    @pytest.mark.parametrize("decode_times,time_vars", [(False, []), (True, ["time"])])
    def test_kerchunk_roundtrip_concat(self, tmpdir, format, decode_times, time_vars):
        # set up example xarray dataset
        ds = xr.tutorial.open_dataset("air_temperature", decode_times=decode_times)

        # split into two datasets
        ds1, ds2 = ds.isel(time=slice(None, 1460)), ds.isel(time=slice(1460, None))

        # save it to disk as netCDF (in temporary directory)
        ds1.to_netcdf(f"{tmpdir}/air1.nc")
        ds2.to_netcdf(f"{tmpdir}/air2.nc")

        # use open_dataset_via_kerchunk to read it as references
        vds1 = open_virtual_dataset(
            f"{tmpdir}/air1.nc",
            indexes={},
            loadable_variables=time_vars,
            cftime_variables=time_vars,
        )
        vds2 = open_virtual_dataset(
            f"{tmpdir}/air2.nc",
            indexes={},
            loadable_variables=time_vars,
            cftime_variables=time_vars,
        )

        if decode_times is False:
            assert vds1.time.dtype == np.dtype("float32")
        else:
            assert vds1.time.dtype == np.dtype("<M8[ns]")
            assert "units" in vds1.time.encoding
            assert "calendar" in vds1.time.encoding

        # concatenate virtually along time
        vds = xr.concat([vds1, vds2], dim="time", coords="minimal", compat="override")

        if format == "dict":
            # write those references to an in-memory kerchunk-formatted references dictionary
            ds_refs = vds.virtualize.to_kerchunk(format=format)

            # use fsspec to read the dataset from the kerchunk references dict
            roundtrip = xr.open_dataset(
                ds_refs, engine="kerchunk", decode_times=decode_times
            )
        else:
            # write those references to disk as kerchunk references format
            vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

            # use fsspec to read the dataset from disk via the kerchunk references
            roundtrip = xr.open_dataset(
                f"{tmpdir}/refs.{format}", engine="kerchunk", decode_times=decode_times
            )
        if decode_times is False:
            # assert identical to original dataset
            xrt.assert_identical(roundtrip, ds)
        else:
            # they are very very close! But assert_allclose doesn't seem to work on datetimes
            assert (roundtrip.time - ds.time).sum() == 0
            assert roundtrip.time.dtype == ds.time.dtype
            assert roundtrip.time.encoding["units"] == ds.time.encoding["units"]
            assert roundtrip.time.encoding["calendar"] == ds.time.encoding["calendar"]

    def test_non_dimension_coordinates(self, tmpdir, format):
        # regression test for GH issue #105

        # set up example xarray dataset containing non-dimension coordinate variables
        ds = xr.Dataset(coords={"lat": (["x", "y"], np.arange(6).reshape(2, 3))})

        # save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(f"{tmpdir}/non_dim_coords.nc")

        vds = open_virtual_dataset(f"{tmpdir}/non_dim_coords.nc", indexes={})

        assert "lat" in vds.coords
        assert "coordinates" not in vds.attrs

        if format == "dict":
            # write those references to an in-memory kerchunk-formatted references dictionary
            ds_refs = vds.virtualize.to_kerchunk(format=format)

            # use fsspec to read the dataset from the kerchunk references dict
            roundtrip = xr.open_dataset(ds_refs, engine="kerchunk", decode_times=False)
        else:
            # write those references to disk as kerchunk references format
            vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

            # use fsspec to read the dataset from disk via the kerchunk references
            roundtrip = xr.open_dataset(
                f"{tmpdir}/refs.{format}", engine="kerchunk", decode_times=False
            )

        # assert equal to original dataset
        xrt.assert_identical(roundtrip, ds)


def test_open_scalar_variable(tmpdir):
    # regression test for GH issue #100

    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")

    vds = open_virtual_dataset(f"{tmpdir}/scalar.nc", indexes={})
    assert vds["a"].shape == ()


def test_rechunk_via_manifest_vs_xarray(netcdf3_file, tmpdir):
    # start with uncompressed netCDF3 file on disk

    new_chunks = {"time": 1460}

    # open and split into two time chunks using virtualizarr
    vds = open_virtual_dataset(netcdf3_file, indexes={}).chunk(**new_chunks)

    # check we have not accidentally created a dask array
    assert isinstance(vds["air"].data, ManifestArray)

    refs_path = f"{tmpdir}/refs.json"
    vds.virtualize.to_kerchunk(refs_path, format="json")

    # open chunked references
    roundtrip_ds = xr.open_dataset(
        refs_path,
        engine="kerchunk",
        chunks={},
    )

    # open original file using xarray and dask, and rechunk
    expected_ds = xr.open_dataset(netcdf3_file, chunks={}).chunk(**new_chunks)

    # assert both approaches equivalent
    xrt.assert_identical(roundtrip_ds, expected_ds)
