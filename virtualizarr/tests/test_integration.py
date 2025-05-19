from collections.abc import Mapping
from os.path import relpath
from pathlib import Path
from typing import Any, Callable, Concatenate, TypeAlias

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt

from conftest import ARRAYBYTES_CODEC, ZLIB_CODEC
from virtualizarr import open_virtual_dataset
from virtualizarr.backends import ZarrBackend
from virtualizarr.backends.hdf import HDFBackend
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.tests import (
    has_fastparquet,
    has_icechunk,
    has_kerchunk,
    requires_kerchunk,
    requires_zarr_python,
)
from virtualizarr.tests.utils import obstore_local
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs

RoundtripFunction: TypeAlias = Callable[
    Concatenate[xr.Dataset | xr.DataTree, Path, ...], xr.Dataset | xr.DataTree
]


def test_kerchunk_roundtrip_in_memory_no_concat(array_v3_metadata):
    # Set up example xarray dataset
    chunks_dict = {
        "0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        "0.1": {"path": "/foo.nc", "offset": 200, "length": 100},
    }
    manifest = ChunkManifest(entries=chunks_dict)
    metadata = create_v3_array_metadata(
        shape=(2, 4),
        chunk_shape=(2, 4),
        data_type=np.dtype("float32"),
    )
    marr = ManifestArray(
        metadata=metadata,
        chunkmanifest=manifest,
    )
    vds = xr.Dataset({"a": (["x", "y"], marr)})

    # Use accessor to write it out to kerchunk reference dict
    ds_refs = vds.virtualize.to_kerchunk(format="dict")

    # reconstruct the dataset
    manifest_store = manifeststore_from_kerchunk_refs(ds_refs)
    roundtrip = manifest_store.to_virtual_dataset(loadable_variables=[])

    # # Assert equal to original dataset
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
            marks=pytest.mark.skip(reason="slow"),
        ),
    ],
)
def test_numpy_arrays_to_inlined_kerchunk_refs(
    netcdf4_file, inline_threshold, vars_to_inline,
):
    from kerchunk.hdf import SingleHdf5ToZarr

    # inline_threshold is chosen to test inlining only the variables listed in vars_to_inline
    expected = SingleHdf5ToZarr(
        netcdf4_file, inline_threshold=int(inline_threshold)
    ).translate()
    
    # loading the variables should produce same result as inlining them using kerchunk
    store = obstore_local(netcdf4_file)
    backend = HDFBackend()
    with open_virtual_dataset(
        filepath=netcdf4_file,
        object_reader=store,
        backend=backend,
        loadable_variables=vars_to_inline, 
    ) as vds:
        refs = vds.virtualize.to_kerchunk(format="dict")

        # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/zarr-developers/VirtualiZarr/pull/73#issuecomment-2040931202
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


def roundtrip_as_in_memory_icechunk(
    vdata: xr.Dataset | xr.DataTree,
    tmp_path: Path,
    virtualize_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> xr.Dataset | xr.DataTree:
    from icechunk import Repository, Storage

    # create an in-memory icechunk store
    storage = Storage.new_in_memory()
    repo = Repository.create(storage=storage)
    session = repo.writable_session("main")

    # write those references to an icechunk store
    vdata.virtualize.to_icechunk(session.store, **(virtualize_kwargs or {}))

    if isinstance(vdata, xr.DataTree):
        # read the dataset from icechunk
        return xr.open_datatree(
            session.store,  # type: ignore
            engine="zarr",
            zarr_format=3,
            consolidated=False,
            **kwargs,
        )

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
    def test_zarr_roundtrip(
        self,
        tmp_path,
        roundtrip_func: RoundtripFunction,
    ):
        air_zarr_path = str(tmp_path / "air_temperature.zarr")
        store = obstore_local(filepath=air_zarr_path)
        backend = ZarrBackend()
        with xr.tutorial.open_dataset("air_temperature", decode_times=False) as ds:
            # TODO: for now we will save as Zarr V3. Later we can parameterize it for V2.
            ds.to_zarr(air_zarr_path, zarr_format=3, consolidated=False)
            with open_virtual_dataset(
                filepath=air_zarr_path,
                object_reader=store,
                backend=backend,
            ) as vds:
                roundtrip = roundtrip_func(vds, tmp_path, decode_times=False)

                # assert all_close to original dataset
                xrt.assert_allclose(roundtrip, ds)

                # assert coordinate attributes are maintained
                for coord in ds.coords:
                    assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    def test_roundtrip_no_concat(
        self,
        tmp_path,
        roundtrip_func: RoundtripFunction,
    ):
        air_nc_path = str(tmp_path / "air.nc")

        # set up example xarray dataset
        with xr.tutorial.open_dataset("air_temperature", decode_times=False) as ds:
            # save it to disk as netCDF (in temporary directory)
            ds.to_netcdf(air_nc_path)
            store = obstore_local(air_nc_path)
            backend = HDFBackend()
            # use open_dataset_via_kerchunk to read it as references
            with open_virtual_dataset(
                filepath=air_nc_path,
                object_reader=store,
                backend=backend
            ) as vds:
                roundtrip = roundtrip_func(vds, tmp_path, decode_times=False)
                # assert all_close to original dataset
                xrt.assert_allclose(roundtrip, ds)

                # TODO fails with ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                # assert ds["air"].attrs == roundtrip["air"].attrs

                # assert coordinate attributes are maintained
                for coord in ds.coords:
                    assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    @pytest.mark.parametrize("decode_times,time_vars", [(False, []), (True, ["time"])])
    def test_kerchunk_roundtrip_concat(
        self,
        tmp_path: Path,
        roundtrip_func: RoundtripFunction,
        decode_times: bool,
        time_vars: list[str],
    ):
        # set up example xarray dataset
        with xr.tutorial.open_dataset(
            "air_temperature", decode_times=decode_times
        ) as ds:
            # split into two datasets
            ds1 = ds.isel(time=slice(None, 1460))
            ds2 = ds.isel(time=slice(1460, None))

            # save it to disk as netCDF (in temporary directory)
            air1_nc_path = str(tmp_path / "air1.nc")
            air2_nc_path = str(tmp_path / "air2.nc")
            ds1.to_netcdf(air1_nc_path)
            ds2.to_netcdf(air2_nc_path)

            # use open_dataset_via_kerchunk to read it as references
            backend = HDFBackend()
            store1 = obstore_local(air1_nc_path) 
            store2 = obstore_local(air2_nc_path) 
            with (
                open_virtual_dataset(
                    filepath=air1_nc_path,
                    object_reader=store1,
                    backend=backend,
                    loadable_variables=time_vars,
                ) as vds1,
                open_virtual_dataset(
                    filepath=air2_nc_path,
                    object_reader=store2,
                    backend=backend,
                    loadable_variables=time_vars,
                ) as vds2,
            ):
                if not decode_times:
                    assert vds1.time.dtype == np.dtype("float32")
                else:
                    assert vds1.time.dtype == np.dtype("<M8[ns]")
                    assert "units" in vds1.time.encoding
                    assert "calendar" in vds1.time.encoding

                # concatenate virtually along time
                vds = xr.concat(
                    [vds1, vds2], dim="time", coords="minimal", compat="override"
                )

                roundtrip = roundtrip_func(vds, tmp_path, decode_times=decode_times)

                # assert all_close to original dataset
                xrt.assert_allclose(roundtrip, ds)

                # assert coordinate attributes are maintained
                for coord in ds.coords:
                    assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

                if decode_times:
                    assert roundtrip.time.dtype == ds.time.dtype
                    assert roundtrip.time.encoding["units"] == ds.time.encoding["units"]
                    assert (
                        roundtrip.time.encoding["calendar"]
                        == ds.time.encoding["calendar"]
                    )

    @pytest.mark.xfail(
        reason="To fix coordinate behavior with HDF reader"
    )
    def test_non_dimension_coordinates(
        self,
        tmp_path: Path,
        roundtrip_func: RoundtripFunction,
    ):
        # regression test for GH issue #105

        # set up example xarray dataset containing non-dimension coordinate variables
        ds = xr.Dataset(coords={"lat": (["x", "y"], np.arange(6.0).reshape(2, 3))})

        # save it to disk as netCDF (in temporary directory)
        nc_path = str(tmp_path / "non_dim_coords.nc")
        ds.to_netcdf(nc_path)

        store = obstore_local(nc_path)
        backend = HDFBackend()
        with open_virtual_dataset(
            filepath=nc_path, 
            object_reader=store,
            backend=backend
        ) as vds:
            assert "lat" in vds.coords
            assert "coordinates" not in vds.attrs

            roundtrip = roundtrip_func(vds, tmp_path)

            # # assert equal to original dataset
            xrt.assert_allclose(roundtrip, ds)

            # # assert coordinate attributes are maintained
            for coord in ds.coords:
                assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    @pytest.mark.xfail(
        reason="Datetime and timedelta data types not yet supported by zarr-python 3.0"  # https://github.com/zarr-developers/zarr-python/issues/2616
    )
    def test_datetime64_dtype_fill_value(
        self, tmpdir, roundtrip_func, array_v3_metadata
    ):
        chunks_dict = {
            "0.0.0": {"path": "/foo.nc", "offset": 100, "length": 100},
        }
        manifest = ChunkManifest(entries=chunks_dict)
        chunks = (1, 1, 1)
        shape = (1, 1, 1)
        metadata = array_v3_metadata(
            shape=shape,
            chunks=chunks,
            codecs=[ARRAYBYTES_CODEC, ZLIB_CODEC],
            data_type=np.dtype("M8[ns]"),
        )
        marr1 = ManifestArray(metadata=metadata, chunkmanifest=manifest)
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


@pytest.mark.parametrize(
    "roundtrip_func", [roundtrip_as_in_memory_icechunk] if has_icechunk else []
)
@pytest.mark.parametrize("decode_times", (False, True))
@pytest.mark.parametrize("time_vars", ([], ["time"]))
@pytest.mark.parametrize("inherit", (False, True))
def test_datatree_roundtrip(
    tmp_path: Path,
    roundtrip_func: RoundtripFunction,
    decode_times: bool,
    time_vars: list[str],
    inherit: bool,
):
    # set up example xarray dataset
    with xr.tutorial.open_dataset("air_temperature", decode_times=decode_times) as ds:
        # split into two datasets
        ds1 = ds.isel(time=slice(None, 1460))
        ds2 = ds.isel(time=slice(1460, None))

        # save it to disk as netCDF (in temporary directory)
        air1_nc_path = str(tmp_path / "air1.nc")
        air2_nc_path = str(tmp_path / "air2.nc")
        ds1.to_netcdf(air1_nc_path)
        ds2.to_netcdf(air2_nc_path)

        store1 = obstore_local(filepath=air1_nc_path)
        store2 = obstore_local(filepath=air2_nc_path)
        backend = HDFBackend()
        # use open_dataset_via_kerchunk to read it as references
        with (
            open_virtual_dataset(
                filepath=air1_nc_path,
                object_reader=store1,
                backend=backend,
                loadable_variables=time_vars,
                decode_times=decode_times,
            ) as vds1,
            open_virtual_dataset(
                filepath=air2_nc_path,
                object_reader=store2,
                backend=backend,
                loadable_variables=time_vars,
                decode_times=decode_times,
            ) as vds2,
        ):
            if not decode_times or not time_vars:
                assert vds1.time.dtype == np.dtype("float32")
                assert vds2.time.dtype == np.dtype("float32")
            else:
                assert vds1.time.dtype == np.dtype("<M8[ns]")
                assert vds2.time.dtype == np.dtype("<M8[ns]")
                assert "units" in vds1.time.encoding
                assert "units" in vds2.time.encoding
                assert "calendar" in vds1.time.encoding
                assert "calendar" in vds2.time.encoding

            vdt = xr.DataTree.from_dict({"/vds1": vds1, "/nested/vds2": vds2})

            with roundtrip_func(
                vdt,
                tmp_path,
                virtualize_kwargs=dict(write_inherited_coords=inherit),
                decode_times=decode_times,
            ) as roundtrip:
                assert isinstance(roundtrip, xr.DataTree)

                # assert all_close to original dataset
                roundtrip_vds1 = roundtrip["/vds1"].to_dataset()
                roundtrip_vds2 = roundtrip["/nested/vds2"].to_dataset()
                xrt.assert_allclose(roundtrip_vds1, ds1)
                xrt.assert_allclose(roundtrip_vds2, ds2)

                # assert coordinate attributes are maintained
                for coord in ds1.coords:
                    assert ds1.coords[coord].attrs == roundtrip_vds1.coords[coord].attrs
                for coord in ds2.coords:
                    assert ds2.coords[coord].attrs == roundtrip_vds2.coords[coord].attrs

                if decode_times:
                    assert roundtrip_vds1.time.dtype == ds1.time.dtype
                    assert roundtrip_vds2.time.dtype == ds2.time.dtype
                    assert (
                        roundtrip_vds1.time.encoding["units"]
                        == ds1.time.encoding["units"]
                    )
                    assert (
                        roundtrip_vds2.time.encoding["units"]
                        == ds2.time.encoding["units"]
                    )
                    assert (
                        roundtrip_vds1.time.encoding["calendar"]
                        == ds1.time.encoding["calendar"]
                    )
                    assert (
                        roundtrip_vds2.time.encoding["calendar"]
                        == ds2.time.encoding["calendar"]
                    )


def test_open_scalar_variable(tmp_path: Path):
    # regression test for GH issue #100

    nc_path = str(tmp_path / "scalar.nc")
    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(nc_path)

    store = obstore_local(nc_path)
    backend = HDFBackend()
    with open_virtual_dataset(
        filepath=nc_path,
        object_reader=store,
        backend=backend,
    ) as vds:
        assert vds["a"].shape == ()


class TestPathsToURIs:
    def test_convert_absolute_paths_to_uris(self, netcdf4_file):
        store = obstore_local(filepath=netcdf4_file)
        backend = HDFBackend()
        with open_virtual_dataset(
            filepath=netcdf4_file,
            object_reader=store,
            backend=backend,
        ) as vds:
            expected_path = Path(netcdf4_file).as_uri()
            manifest = vds["air"].data.manifest.dict()
            path = manifest["0.0.0"]["path"]

            assert path == expected_path

    def test_convert_relative_paths_to_uris(self, netcdf4_file):
        relative_path = relpath(netcdf4_file)
        store = obstore_local(relative_path)
        backend = HDFBackend()
        with open_virtual_dataset(
            filepath=relative_path,
            object_reader=store,
            backend=backend,
        ) as vds:
            expected_path = Path(netcdf4_file).as_uri()
            manifest = vds["air"].data.manifest.dict()
            path = manifest["0.0.0"]["path"]

            assert path == expected_path
