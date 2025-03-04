from os.path import relpath
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import (
    has_fastparquet,
    has_icechunk,
    has_kerchunk,
    parametrize_over_hdf_backends,
    requires_kerchunk,
    requires_zarr_python,
)
from virtualizarr.translators.kerchunk import (
    dataset_from_kerchunk_refs,
)
from virtualizarr.zarr import ZArray


def test_kerchunk_roundtrip_in_memory_no_concat():
    # Set up example xarray dataset
    chunks_dict = {
        "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    marr = ManifestArray(
        zarray=dict(
            shape=(2, 4),
            dtype=np.dtype("<i8"),
            chunks=(2, 2),
            compressor=None,
            filters=None,
            fill_value=None,
            order="C",
        ),
        chunkmanifest=manifest,
    )
    vds = xr.Dataset({"a": (["x", "y"], marr)})

    # Use accessor to write it out to kerchunk reference dict
    ds_refs = vds.virtualize.to_kerchunk(format="dict")

    # Use dataset_from_kerchunk_refs to reconstruct the dataset
    roundtrip = dataset_from_kerchunk_refs(ds_refs)

    # Assert equal to original dataset
    xrt.assert_equal(roundtrip, vds)


@requires_kerchunk
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
@parametrize_over_hdf_backends
def test_numpy_arrays_to_inlined_kerchunk_refs(
    netcdf4_file, inline_threshold, vars_to_inline, hdf_backend
):
    from kerchunk.hdf import SingleHdf5ToZarr

    # inline_threshold is chosen to test inlining only the variables listed in vars_to_inline
    expected = SingleHdf5ToZarr(
        netcdf4_file, inline_threshold=int(inline_threshold)
    ).translate()

    # loading the variables should produce same result as inlining them using kerchunk
    vds = open_virtual_dataset(
        netcdf4_file, loadable_variables=vars_to_inline, indexes={}, backend=hdf_backend
    )
    refs = vds.virtualize.to_kerchunk(format="dict")

    # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/TomNicholas/VirtualiZarr/pull/73#issuecomment-2040931202
    # assert refs == expected
    assert refs["refs"]["air/0.0.0"] == expected["refs"]["air/0.0.0"]
    assert refs["refs"]["lon/0"] == expected["refs"]["lon/0"]
    assert refs["refs"]["lat/0"] == expected["refs"]["lat/0"]
    assert refs["refs"]["time/0"] == expected["refs"]["time/0"]


def roundtrip_as_kerchunk_dict(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to an in-memory kerchunk-formatted references dictionary
    ds_refs = vds.virtualize.to_kerchunk(format="dict")

    # use fsspec to read the dataset from the kerchunk references dict
    return xr.open_dataset(ds_refs, engine="kerchunk", **kwargs)


def roundtrip_as_kerchunk_json(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to disk as kerchunk references format
    vds.virtualize.to_kerchunk(f"{tmpdir}/refs.json", format="json")

    # use fsspec to read the dataset from disk via the kerchunk references
    return xr.open_dataset(f"{tmpdir}/refs.json", engine="kerchunk", **kwargs)


def roundtrip_as_kerchunk_parquet(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to disk as kerchunk references format
    vds.virtualize.to_kerchunk(f"{tmpdir}/refs.parquet", format="parquet")

    # use fsspec to read the dataset from disk via the kerchunk references
    return xr.open_dataset(f"{tmpdir}/refs.parquet", engine="kerchunk", **kwargs)


def roundtrip_as_in_memory_icechunk(vds: xr.Dataset, tmpdir, **kwargs):
    from icechunk import Repository, Storage

    # create an in-memory icechunk store
    storage = Storage.new_in_memory()
    repo = Repository.create(storage=storage)
    session = repo.writable_session("main")

    # write those references to an icechunk store
    vds.virtualize.to_icechunk(session.store)

    # read the dataset from icechunk
    return xr.open_zarr(session.store, zarr_format=3, consolidated=False, **kwargs)


@requires_zarr_python
@pytest.mark.parametrize(
    "roundtrip_func",
    [
        *(
            [roundtrip_as_kerchunk_dict, roundtrip_as_kerchunk_json]
            if has_kerchunk
            else []
        ),
        *([roundtrip_as_kerchunk_parquet] if has_kerchunk and has_fastparquet else []),
        *([roundtrip_as_in_memory_icechunk] if has_icechunk else []),
    ],
)
class TestRoundtrip:
    @parametrize_over_hdf_backends
    def test_roundtrip_no_concat(self, tmpdir, roundtrip_func, hdf_backend):
        # set up example xarray dataset
        ds = xr.tutorial.open_dataset("air_temperature", decode_times=False)

        # save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(f"{tmpdir}/air.nc")

        # use open_dataset_via_kerchunk to read it as references
        vds = open_virtual_dataset(f"{tmpdir}/air.nc", indexes={}, backend=hdf_backend)

        roundtrip = roundtrip_func(vds, tmpdir, decode_times=False)

        # assert all_close to original dataset
        xrt.assert_allclose(roundtrip, ds)

        # assert coordinate attributes are maintained
        for coord in ds.coords:
            assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    @parametrize_over_hdf_backends
    @pytest.mark.parametrize("decode_times,time_vars", [(False, []), (True, ["time"])])
    def test_kerchunk_roundtrip_concat(
        self, tmpdir, roundtrip_func, hdf_backend, decode_times, time_vars
    ):
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
            backend=hdf_backend,
        )
        vds2 = open_virtual_dataset(
            f"{tmpdir}/air2.nc",
            indexes={},
            loadable_variables=time_vars,
            backend=hdf_backend,
        )

        if decode_times is False:
            assert vds1.time.dtype == np.dtype("float32")
        else:
            assert vds1.time.dtype == np.dtype("<M8[ns]")
            assert "units" in vds1.time.encoding
            assert "calendar" in vds1.time.encoding

        # concatenate virtually along time
        vds = xr.concat([vds1, vds2], dim="time", coords="minimal", compat="override")

        roundtrip = roundtrip_func(vds, tmpdir, decode_times=decode_times)

        if decode_times is False:
            # assert all_close to original dataset
            xrt.assert_allclose(roundtrip, ds)

            # assert coordinate attributes are maintained
            for coord in ds.coords:
                assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs
        else:
            # they are very very close! But assert_allclose doesn't seem to work on datetimes
            assert (roundtrip.time - ds.time).sum() == 0
            assert roundtrip.time.dtype == ds.time.dtype
            assert roundtrip.time.encoding["units"] == ds.time.encoding["units"]
            assert roundtrip.time.encoding["calendar"] == ds.time.encoding["calendar"]

    @parametrize_over_hdf_backends
    def test_non_dimension_coordinates(self, tmpdir, roundtrip_func, hdf_backend):
        # regression test for GH issue #105

        if hdf_backend:
            pytest.xfail("To fix coordinate behavior with HDF reader")

        # set up example xarray dataset containing non-dimension coordinate variables
        ds = xr.Dataset(coords={"lat": (["x", "y"], np.arange(6.0).reshape(2, 3))})

        # save it to disk as netCDF (in temporary directory)
        ds.to_netcdf(f"{tmpdir}/non_dim_coords.nc")

        vds = open_virtual_dataset(
            f"{tmpdir}/non_dim_coords.nc", indexes={}, backend=hdf_backend
        )
        assert "lat" in vds.coords
        assert "coordinates" not in vds.attrs

        roundtrip = roundtrip_func(vds, tmpdir)

        # assert equal to original dataset
        xrt.assert_allclose(roundtrip, ds)

        # assert coordinate attributes are maintained
        for coord in ds.coords:
            assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    def test_datetime64_dtype_fill_value(self, tmpdir, roundtrip_func):
        if "icechunk" in roundtrip_func.__name__:
            pytest.xfail(
                "zarr 3 does not support datetimes yet "
                "https://github.com/zarr-developers/zarr-python/issues/2616"
            )

        chunks_dict = {
            "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (1, 1, 1)
        shape = (1, 1, 1)
        zarray = ZArray(
            chunks=chunks,
            compressor={"id": "zlib", "level": 1},
            dtype=np.dtype("<M8[ns]"),
            # fill_value=0.0,
            filters=None,
            order="C",
            shape=shape,
            zarr_format=2,
        )
        marr1 = ManifestArray(zarray=zarray, chunkmanifest=manifest)
        vds = xr.Dataset(
            {
                "a": xr.DataArray(
                    marr1,
                    attrs={
                        "_FillValue": np.datetime64("1970-01-01T00:00:00.000000000")
                    },
                )
            }
        )

        roundtrip = roundtrip_func(vds, tmpdir)

        assert roundtrip.a.attrs == vds.a.attrs


@parametrize_over_hdf_backends
def test_open_scalar_variable(tmpdir, hdf_backend):
    # regression test for GH issue #100

    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(f"{tmpdir}/scalar.nc")

    vds = open_virtual_dataset(f"{tmpdir}/scalar.nc", indexes={}, backend=hdf_backend)
    assert vds["a"].shape == ()


@parametrize_over_hdf_backends
class TestPathsToURIs:
    def test_convert_absolute_paths_to_uris(self, netcdf4_file, hdf_backend):
        vds = open_virtual_dataset(netcdf4_file, indexes={}, backend=hdf_backend)

        expected_path = Path(netcdf4_file).as_uri()

        manifest = vds["air"].data.manifest.dict()
        path = manifest["0.0.0"]["path"]
        assert path == expected_path

    def test_convert_relative_paths_to_uris(self, netcdf4_file, hdf_backend):
        relative_path = relpath(netcdf4_file)
        vds = open_virtual_dataset(relative_path, indexes={}, backend=hdf_backend)

        expected_path = Path(netcdf4_file).as_uri()

        manifest = vds["air"].data.manifest.dict()
        path = manifest["0.0.0"]["path"]
        assert path == expected_path
