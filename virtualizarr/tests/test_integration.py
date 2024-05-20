import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset


@pytest.mark.parametrize(
    "inline_threshold, vars_to_inline",
    [
        (5e2, ["lat", "lon"]),
        pytest.param(
            5e4, ["lat", "lon", "time"], marks=pytest.mark.xfail(reason="time encoding")
        ),
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


@pytest.mark.parametrize("format", ["json", "parquet"])
class TestKerchunkRoundtrip:
    def test_kerchunk_roundtrip_no_concat(self, tmpdir, format):
        # set up example xarray dataset
        ds = xr.tutorial.open_dataset("air_temperature", decode_times=False)

        # save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(f"{tmpdir}/air.nc")

        # use open_dataset_via_kerchunk to read it as references
        vds = open_virtual_dataset(f"{tmpdir}/air.nc", indexes={})

        # write those references to disk as kerchunk json
        vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

        # use fsspec to read the dataset from disk via the zarr store
        roundtrip = xr.open_dataset(f"{tmpdir}/refs.{format}", engine="kerchunk")

        # assert equal to original dataset
        xrt.assert_equal(roundtrip, ds)

    def test_kerchunk_roundtrip_concat(self, tmpdir, format):
        # set up example xarray dataset
        ds = xr.tutorial.open_dataset(
            "air_temperature", decode_times=True
        )  # .isel(time=slice(None, 2000))
        del ds.time.encoding["units"]

        # split into two datasets
        ds1, ds2 = ds.isel(time=slice(None, 1460)), ds.isel(time=slice(1460, None))

        # save it to disk as netCDF (in temporary directory)
        ds1.to_netcdf(f"{tmpdir}/air1.nc")
        ds2.to_netcdf(f"{tmpdir}/air2.nc")

        # use open_dataset_via_kerchunk to read it as references
        vds1 = open_virtual_dataset(
            f"{tmpdir}/air1.nc",
            indexes={},
            loadable_variables=["time"],
            cftime_variables=["time"],
        )
        vds2 = open_virtual_dataset(
            f"{tmpdir}/air2.nc",
            indexes={},
            loadable_variables=["time"],
            cftime_variables=["time"],
        )

        # concatenate virtually along time
        vds = xr.concat([vds1, vds2], dim="time", coords="minimal", compat="override")

        # write those references to disk as kerchunk json
        vds.virtualize.to_kerchunk(f"{tmpdir}/refs.{format}", format=format)

        # use fsspec to read the dataset from disk via the zarr store
        roundtrip = xr.open_dataset(f"{tmpdir}/refs.{format}", engine="kerchunk")

        # assert equal to original dataset
        xrt.assert_equal(roundtrip, ds)


def test_open_scalar_variable(tmpdir):
    # regression test for GH issue #100

    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")

    vds = open_virtual_dataset(f"{tmpdir}/scalar.nc")
    assert vds["a"].shape == ()
