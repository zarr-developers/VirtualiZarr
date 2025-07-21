from collections.abc import Mapping
from os.path import relpath
from pathlib import Path
from typing import Any, Callable, Concatenate, TypeAlias

import numpy as np
import pytest
import xarray as xr
import xarray.testing as xrt
from obstore.store import LocalStore, from_url

from conftest import ARRAYBYTES_CODEC, ZLIB_CODEC
from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers import HDFParser, ZarrParser
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.tests import (
    has_fastparquet,
    has_icechunk,
    has_kerchunk,
    requires_kerchunk,
    requires_network,
    requires_zarr_python,
    slow_test,
)
from virtualizarr.tests.utils import PYTEST_TMP_DIRECTORY_URL_PREFIX

icechunk = pytest.importorskip("icechunk")

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
    ds_refs = vds.vz.to_kerchunk(format="dict")

    # reconstruct the dataset
    manifestgroup = manifestgroup_from_kerchunk_refs(ds_refs)
    manifeststore = ManifestStore(group=manifestgroup)
    roundtrip = manifeststore.to_virtual_dataset(loadable_variables=[])

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
            marks=pytest.mark.skip(reason="slow"),
        ),
    ],
)
def test_numpy_arrays_to_inlined_kerchunk_refs(
    netcdf4_file, inline_threshold, vars_to_inline, local_registry
):
    from kerchunk.hdf import SingleHdf5ToZarr

    # inline_threshold is chosen to test inlining only the variables listed in vars_to_inline
    expected = SingleHdf5ToZarr(
        netcdf4_file, inline_threshold=int(inline_threshold)
    ).translate()

    # loading the variables should produce same result as inlining them using kerchunk
    parser = HDFParser()
    with open_virtual_dataset(
        url=netcdf4_file,
        registry=local_registry,
        parser=parser,
        loadable_variables=vars_to_inline,
    ) as vds:
        refs = vds.vz.to_kerchunk(format="dict")

        # TODO I would just compare the entire dicts but kerchunk returns inconsistent results - see https://github.com/zarr-developers/VirtualiZarr/pull/73#issuecomment-2040931202
        # assert refs == expected
        assert refs["refs"]["air/0.0.0"] == expected["refs"]["air/0.0.0"]
        assert refs["refs"]["lon/0"] == expected["refs"]["lon/0"]
        assert refs["refs"]["lat/0"] == expected["refs"]["lat/0"]
        assert refs["refs"]["time/0"] == expected["refs"]["time/0"]


def roundtrip_as_kerchunk_dict(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to an in-memory kerchunk-formatted references dictionary
    ds_refs = vds.vz.to_kerchunk(format="dict")

    # use fsspec to read the dataset from the kerchunk references dict
    return xr.open_dataset(ds_refs, engine="kerchunk", **kwargs)


def roundtrip_as_kerchunk_json(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to disk as kerchunk references format
    vds.vz.to_kerchunk(f"{tmpdir}/refs.json", format="json")

    # use fsspec to read the dataset from disk via the kerchunk references
    return xr.open_dataset(f"{tmpdir}/refs.json", engine="kerchunk", **kwargs)


def roundtrip_as_kerchunk_parquet(vds: xr.Dataset, tmpdir, **kwargs):
    # write those references to disk as kerchunk references format
    vds.vz.to_kerchunk(f"{tmpdir}/refs.parquet", format="parquet")

    # use fsspec to read the dataset from disk via the kerchunk references
    return xr.open_dataset(f"{tmpdir}/refs.parquet", engine="kerchunk", **kwargs)


def roundtrip_as_in_memory_icechunk(
    vdata: xr.Dataset | xr.DataTree,
    tmp_path: Path,
    virtualize_kwargs: Mapping[str, Any] | None = None,
    **kwargs,
) -> xr.Dataset | xr.DataTree:
    # create an in-memory icechunk store
    storage = icechunk.Storage.new_in_memory()

    config = icechunk.RepositoryConfig.default()

    container = icechunk.VirtualChunkContainer(
        url_prefix=PYTEST_TMP_DIRECTORY_URL_PREFIX,
        store=icechunk.local_filesystem_store(PYTEST_TMP_DIRECTORY_URL_PREFIX),
    )
    config.set_virtual_chunk_container(container)

    repo = icechunk.Repository.create(
        storage=storage,
        config=config,
        authorize_virtual_chunk_access={PYTEST_TMP_DIRECTORY_URL_PREFIX: None},
    )
    session = repo.writable_session("main")

    # write those references to an icechunk store
    vdata.vz.to_icechunk(session.store, **(virtualize_kwargs or {}))
    session.commit("Test")

    read_only_session = repo.readonly_session("main")

    if isinstance(vdata, xr.DataTree):
        # read the dataset from icechunk
        return xr.open_datatree(
            read_only_session.store,  # type: ignore
            engine="zarr",
            zarr_format=3,
            consolidated=False,
            **kwargs,
        )

    # read the dataset from icechunk
    return xr.open_zarr(
        read_only_session.store, zarr_format=3, consolidated=False, **kwargs
    )


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
        air_zarr_url = f"file://{air_zarr_path}"
        with xr.tutorial.open_dataset("air_temperature", decode_times=False) as ds:
            # TODO: for now we will save as Zarr V3. Later we can parameterize it for V2.
            ds.to_zarr(air_zarr_path, zarr_format=3, consolidated=False)
            store = LocalStore(prefix=air_zarr_path)
            registry = ObjectStoreRegistry({air_zarr_url: store})
            parser = ZarrParser()
            with open_virtual_dataset(
                url=air_zarr_url,
                registry=registry,
                parser=parser,
            ) as vds:
                roundtrip = roundtrip_func(vds, tmp_path, decode_times=False)

                # assert all_close to original dataset
                xrt.assert_allclose(roundtrip, ds)

                # assert coordinate attributes are maintained
                for coord in ds.coords:
                    assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    def test_roundtrip_no_concat(
        self, tmp_path, roundtrip_func: RoundtripFunction, local_registry
    ):
        air_nc_path = str(tmp_path / "air.nc")

        # set up example xarray dataset
        with xr.tutorial.open_dataset("air_temperature", decode_times=False) as ds:
            # save it to disk as netCDF (in temporary directory)
            ds.to_netcdf(air_nc_path)
            parser = HDFParser()
            # use open_dataset_via_kerchunk to read it as references
            with open_virtual_dataset(
                url=air_nc_path, registry=local_registry, parser=parser
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
        local_registry,
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
            parser = HDFParser()
            with (
                open_virtual_dataset(
                    url=air1_nc_path,
                    registry=local_registry,
                    parser=parser,
                    loadable_variables=time_vars,
                ) as vds1,
                open_virtual_dataset(
                    url=air2_nc_path,
                    registry=local_registry,
                    parser=parser,
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

    def test_non_dimension_coordinates(
        self, tmp_path: Path, roundtrip_func: RoundtripFunction, local_registry
    ):
        # regression test for GH issue #105

        # set up example xarray dataset containing non-dimension coordinate variables
        ds = xr.Dataset(coords={"lat": (["x", "y"], np.arange(6.0).reshape(2, 3))})

        # save it to disk as netCDF (in temporary directory)
        nc_path = str(tmp_path / "non_dim_coords.nc")
        ds.to_netcdf(nc_path)

        parser = HDFParser()
        with open_virtual_dataset(
            url=nc_path, registry=local_registry, parser=parser
        ) as vds:
            assert "lat" in vds.coords
            assert "coordinates" not in vds.attrs

            roundtrip = roundtrip_func(vds, tmp_path)

            # assert equal to original dataset
            xrt.assert_allclose(roundtrip, ds)

            # assert coordinate attributes are maintained
            for coord in ds.coords:
                assert ds.coords[coord].attrs == roundtrip.coords[coord].attrs

    def test_datetime64_dtype_fill_value(
        self, tmpdir, roundtrip_func, array_v3_metadata
    ):
        if roundtrip_func == roundtrip_as_in_memory_icechunk:
            pytest.xfail(reason="xarray can't decode the ns datetime fill_value")

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
    local_registry,
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
        parser = HDFParser()
        # use open_dataset_via_kerchunk to read it as references
        with (
            open_virtual_dataset(
                url=air1_nc_path,
                registry=local_registry,
                parser=parser,
                loadable_variables=time_vars,
                decode_times=decode_times,
            ) as vds1,
            open_virtual_dataset(
                url=air2_nc_path,
                registry=local_registry,
                parser=parser,
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


def test_open_scalar_variable(tmp_path: Path, local_registry):
    # regression test for GH issue #100

    nc_path = str(tmp_path / "scalar.nc")
    ds = xr.Dataset(data_vars={"a": 0})
    ds.to_netcdf(nc_path)

    parser = HDFParser()
    with open_virtual_dataset(
        url=nc_path,
        registry=local_registry,
        parser=parser,
    ) as vds:
        assert vds["a"].shape == ()
    ms = parser(url=f"file://{nc_path}", registry=local_registry)
    roundtripped = xr.open_zarr(ms, consolidated=False, zarr_format=3)
    xr.testing.assert_allclose(ds, roundtripped.load())


class TestPathsToURLs:
    def test_convert_absolute_paths_to_urls(self, netcdf4_file, local_registry):
        parser = HDFParser()
        with open_virtual_dataset(
            url=netcdf4_file,
            registry=local_registry,
            parser=parser,
        ) as vds:
            expected_path = Path(netcdf4_file).as_uri()
            manifest = vds["air"].data.manifest.dict()
            path = manifest["0.0.0"]["path"]

            assert path == expected_path

    def test_convert_relative_paths_to_urls(self, netcdf4_file, local_registry):
        relative_path = relpath(netcdf4_file)
        parser = HDFParser()
        with open_virtual_dataset(
            url=relative_path,
            registry=local_registry,
            parser=parser,
        ) as vds:
            expected_path = Path(netcdf4_file).as_uri()
            manifest = vds["air"].data.manifest.dict()
            path = manifest["0.0.0"]["path"]

            assert path == expected_path


@requires_kerchunk
@requires_network
@slow_test
def test_roundtrip_dataset_with_multiple_compressors():
    # Regression test to make sure we can load data with a compression and a shuffle codec
    # TODO: Simplify this test to not require network access
    import s3fs

    bucket = "s3://nex-gddp-cmip6"
    path = "NEX-GDDP-CMIP6/ACCESS-CM2/ssp126/r1i1p1f1/tasmax/tasmax_day_ACCESS-CM2_ssp126_r1i1p1f1_gn_2015_v2.0.nc"
    url = f"{bucket}/{path}"
    store = from_url(bucket, region="us-west-2", skip_signature=True)
    registry = ObjectStoreRegistry({bucket: store})
    parser = HDFParser()
    vds = open_virtual_dataset(
        url=url, parser=parser, registry=registry, loadable_variables=[]
    )

    ds_refs = vds.vz.to_kerchunk(format="dict")
    fs = s3fs.S3FileSystem(anon=True)
    with (
        xr.open_dataset(fs.open(url), engine="h5netcdf", decode_times=True) as expected,
        xr.open_dataset(
            ds_refs,
            decode_times=True,
            engine="kerchunk",
            storage_options={"remote_options": {"anon": True}},
        ) as observed,
    ):
        xr.testing.assert_allclose(expected, observed)
