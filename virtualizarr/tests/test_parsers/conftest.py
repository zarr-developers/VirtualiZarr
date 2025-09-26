import warnings
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset
from packaging.version import Version
from xarray.tests.test_dataset import create_test_data
from xarray.util.print_versions import netcdf_and_hdf5_versions

try:
    import hdf5plugin  # type: ignore
except ModuleNotFoundError:
    hdf5plugin = None  # type: ignore
    warnings.warn("hdf5plugin is required for HDF reader")


@pytest.fixture
def empty_chunks_hdf5_url(tmpdir):
    ds = xr.Dataset({"data": []})
    filepath = f"{tmpdir}/empty_chunks.nc"
    ds.to_netcdf(filepath, engine="h5netcdf")
    return f"file://{filepath}"


@pytest.fixture
def empty_dataset_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/empty_dataset.nc"
    f = h5py.File(filepath, "w")
    f.create_dataset("data", shape=(0,), dtype="f")
    return f"file://{filepath}"


@pytest.fixture
def no_chunks_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/no_chunks.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    return f"file://{filepath}"


@pytest.fixture
def fill_value_scalar_no_chunks_nc4_url(tmpdir):
    filepath = f"{tmpdir}/fill_value_scalar_no_chunks.nc4"
    f = Dataset(filepath, "w")
    f.createVariable("data", "<i4", fill_value=-999)
    f.long_name = "empty scalar data"
    f.close()
    return f"file://{filepath}"


@pytest.fixture
def chunked_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/chunks.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((100, 100))
    f.create_dataset(name="data", data=data, chunks=(50, 50))
    return f"file://{filepath}"


@pytest.fixture
def single_dimension_scale_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/single_dimension_scale.nc"
    f = h5py.File(filepath, "w")
    data = [1, 2]
    x = [0, 1]
    f.create_dataset(name="data", data=data)
    f.create_dataset(name="x", data=x)
    f["x"].make_scale()
    f["data"].dims[0].attach_scale(f["x"])
    return f"file://{filepath}"


@pytest.fixture
def is_scale_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/is_scale.nc"
    f = h5py.File(filepath, "w")
    data = [1, 2]
    f.create_dataset(name="data", data=data)
    f["data"].make_scale()
    return f"file://{filepath}"


@pytest.fixture
def multiple_dimension_scales_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/multiple_dimension_scales.nc"
    f = h5py.File(filepath, "w")
    data = [1, 2]
    f.create_dataset(name="data", data=data)
    f.create_dataset(name="x", data=[0, 1])
    f.create_dataset(name="y", data=[0, 1])
    f["x"].make_scale()
    f["y"].make_scale()
    f["data"].dims[0].attach_scale(f["x"])
    f["data"].dims[0].attach_scale(f["y"])
    return f"file://{filepath}"


@pytest.fixture
def chunked_dimensions_netcdf4_url(tmpdir):
    filepath = f"{tmpdir}/chunks_dimension.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((100, 100))
    x = np.random.random((100))
    y = np.random.random((100))
    f.create_dataset(name="data", data=data, chunks=(50, 50))
    f.create_dataset(name="x", data=x)
    f.create_dataset(name="y", data=y)
    f["data"].dims[0].attach_scale(f["x"])
    f["data"].dims[1].attach_scale(f["y"])
    return f"file://{filepath}"


@pytest.fixture
def string_attributes_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/attributes.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    f["data"].attrs["attribute_name"] = "attribute_name"
    f["data"].attrs["attribute_name2"] = "attribute_name2"
    return f"file://{filepath}"


@pytest.fixture
def root_attributes_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/root_attributes.nc"
    f = h5py.File(filepath, "w")
    f.attrs["attribute_name"] = "attribute_name"
    return f"file://{filepath}"


@pytest.fixture
def group_hdf5_url(tmpdir):
    filepath = f"{tmpdir}/group.nc"
    f = h5py.File(filepath, "w")
    g = f.create_group("group")
    data = np.random.random((10, 10))
    g.create_dataset("data", data=data)
    return f"file://{filepath}"


@pytest.fixture
def nested_group_hdf5_url(tmp_path: Path) -> str:
    filepath = str(tmp_path / "nested_group.nc")

    with h5py.File(filepath, "w") as f:
        g = f.create_group("group")
        data = np.random.random((10, 10))
        g.create_dataset("data", data=data)
        g.create_group("nested_group")

    return f"file://{filepath}"


@pytest.fixture
def multiple_datasets_hdf5_url(tmp_path: Path) -> str:
    filepath = str(tmp_path / "multiple_datasets.nc")

    with h5py.File(filepath, "w") as f:
        data = np.random.random((10, 10))
        f.create_dataset(name="data", data=data, chunks=None)
        f.create_dataset(name="data2", data=data, chunks=None)

    return f"file://{filepath}"


@pytest.fixture
def np_uncompressed():
    return np.arange(100)


@pytest.fixture(params=["gzip", "blosc_lz4", "lz4", "bzip2", "zstd", "shuffle"])
def filter_encoded_hdf5_file(tmp_path: Path, np_uncompressed, request) -> str:
    assert hdf5plugin is not None  # make type-checkers happy
    filepath = str(tmp_path / f"{request.param}.nc")

    with h5py.File(filepath, "w") as f:
        if request.param == "gzip":
            f.create_dataset(
                name="data",
                data=np_uncompressed,
                compression="gzip",
                compression_opts=1,
            )
        if request.param == "blosc_lz4":
            f.create_dataset(
                name="data",
                data=np_uncompressed,
                **hdf5plugin.Blosc(
                    cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
                ),
            )
        if request.param == "lz4":
            f.create_dataset(
                name="data", data=np_uncompressed, **hdf5plugin.LZ4(nbytes=0)
            )
        if request.param == "bzip2":
            f.create_dataset(name="data", data=np_uncompressed, **hdf5plugin.BZip2())
        if request.param == "zstd":
            f.create_dataset(
                name="data", data=np_uncompressed, **hdf5plugin.Zstd(clevel=2)
            )
        if request.param == "shuffle":
            f.create_dataset(name="data", data=np_uncompressed, shuffle=True)

    return filepath


@pytest.fixture(params=["gzip"])
def filter_encoded_roundtrip_hdf5_file(tmp_path: Path, request) -> str:
    with xr.tutorial.open_dataset("air_temperature") as ds:
        encoding = {}
        if request.param == "gzip":
            encoding_config = {"zlib": True, "complevel": 1}

        for var_name in ds.variables:
            encoding[var_name] = encoding_config

        filepath = tmp_path / f"{request.param}_xarray.nc"
        ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)

        return str(filepath)


@pytest.fixture()
def skip_test_for_libhdf5_version():
    versions = netcdf_and_hdf5_versions()
    libhdf5_version = Version(versions[0][1])
    return libhdf5_version < Version("1.14")


@pytest.fixture(params=["blosc_zlib", ""])
def filter_encoded_roundtrip_netcdf4_file(
    tmpdir, request, skip_test_for_libhdf5_version
):
    if skip_test_for_libhdf5_version:
        pytest.skip("Requires libhdf5 >= 1.14")
    ds = create_test_data(dim_sizes=(20, 80, 10))
    encoding_config = {
        "chunksizes": (20, 40),
        "original_shape": ds.var2.shape,
        "blosc_shuffle": 1,
        "fletcher32": False,
    }
    if "blosc" in request.param:
        encoding_config["compression"] = request.param
    #  Check on how handle scalar dim.
    ds = ds.drop_dims("dim3")
    ds["var2"].encoding.update(encoding_config)
    filepath = f"{tmpdir}/{request.param}_xarray.nc"
    ds.to_netcdf(filepath, engine="netcdf4")
    return {
        "filepath": filepath,
        "url": f"file://{filepath}",
        "compressor": request.param,
    }


@pytest.fixture
def np_uncompressed_int16():
    return np.arange(100, dtype=np.int16)


@pytest.fixture
def offset():
    return np.float32(5.0)


@pytest.fixture
def add_offset_hdf5_file(tmp_path: Path, np_uncompressed_int16, offset) -> str:
    filepath = str(tmp_path / "offset.nc")

    with h5py.File(filepath, "w") as f:
        data = np_uncompressed_int16 - offset
        f.create_dataset(name="data", data=data, chunks=True)
        f["data"].attrs.create(name="add_offset", data=offset)

    return filepath


@pytest.fixture
def scale_factor():
    return 0.01


@pytest.fixture
def scale_add_offset_hdf5_file(
    tmp_path: Path, np_uncompressed_int16, offset, scale_factor
) -> str:
    filepath = str(tmp_path / "scale_offset.nc")

    with h5py.File(filepath, "w") as f:
        data = (np_uncompressed_int16 - offset) / scale_factor
        f.create_dataset(name="data", data=data, chunks=True)
        f["data"].attrs.create(name="add_offset", data=offset)
        f["data"].attrs.create(name="scale_factor", data=np.array([scale_factor]))

    return filepath


@pytest.fixture()
def chunked_roundtrip_hdf5_url(tmpdir):
    ds = create_test_data(dim_sizes=(20, 80, 10))
    ds = ds.drop_dims("dim3")
    filepath = f"{tmpdir}/chunked_xarray.nc"
    ds.to_netcdf(
        filepath, engine="netcdf4", encoding={"var2": {"chunksizes": (10, 10)}}
    )
    return f"file://{filepath}"


@pytest.fixture(params=["gzip", "zlib"])
def filter_and_cf_roundtrip_hdf5_file(tmpdir, request):
    x = np.arange(100)
    y = np.arange(100)
    fill_value = np.int16(-9999)
    temperature = 0.1 * x[:, None] + 0.1 * y[None, :]
    temperature[0][0] = fill_value
    ds = xr.Dataset(
        {"temperature": (["x", "y"], temperature)},
        coords={"x": np.arange(100), "y": np.arange(100)},
    )
    encoding = {
        "temperature": {
            "dtype": "int16",
            "scale_factor": 0.1,
            "add_offset": 273.15,
            "_FillValue": fill_value,
        },
        "x": {"_FillValue": fill_value},
        "y": {"_FillValue": fill_value},
    }
    if request.param == "gzip":
        encoding["temperature"]["compression"] = "gzip"
        encoding["temperature"]["compression_opts"] = 7

    if request.param == "zlib":
        encoding["temperature"]["zlib"] = True
        encoding["temperature"]["complevel"] = 9

    from random import randint

    filepath = f"{tmpdir}/{request.param}_{randint(0, 100)}_cf_roundtrip.nc"
    ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)

    return filepath


@pytest.fixture
def root_coordinates_hdf5_file(tmp_path: Path, np_uncompressed_int16) -> str:
    filepath = str(tmp_path / "coordinates.nc")

    with h5py.File(filepath, "w") as f:
        data = np.random.random((100, 100))
        f.create_dataset(name="data", data=data, chunks=True)
        f.create_dataset(name="lat", data=data)
        f.create_dataset(name="lon", data=data)
        f.attrs.create(name="coordinates", data="lat lon")

    return filepath


@pytest.fixture
def netcdf3_file(tmp_path: Path) -> Path:
    ds = xr.Dataset({"foo": ("x", np.array([1, 2, 3]))})

    filepath = tmp_path / "file.nc"
    ds.to_netcdf(filepath, format="NETCDF3_CLASSIC")

    return filepath


@pytest.fixture
def non_coord_dim(tmpdir):
    filepath = f"{tmpdir}/non_coord_dim.nc"
    ds = create_test_data(dim_sizes=(20, 80, 10))
    ds = ds.drop_dims("dim3")
    ds.to_netcdf(filepath, engine="netcdf4")
    return filepath


@pytest.fixture
def scalar_fill_value_hdf5_url(tmp_path: Path) -> str:
    filepath = str(tmp_path / "scalar_fill_value.nc")

    with h5py.File(filepath, "w") as f:
        data = np.random.randint(0, 10, size=(5))
        fill_value = 42
        f.create_dataset(name="data", data=data, chunks=True, fillvalue=fill_value)

    return f"file://{filepath}"


compound_dtype = np.dtype(
    [
        ("id", "i4"),  # 4-byte integer
        ("temperature", "f4"),  # 4-byte float
    ]
)

compound_data = np.array(
    [
        (1, 98.6),
        (2, 101.3),
    ],
    dtype=compound_dtype,
)

compound_fill = (-9999, -9999.0)

fill_values = [
    {"fill_value": -9999, "data": np.random.randint(0, 10, size=(5))},
    {"fill_value": -9999.0, "data": np.random.random(5)},
    {"fill_value": np.nan, "data": np.random.random(5)},
    {"fill_value": False, "data": np.array([True, False, False, True, True])},
    {"fill_value": "NaN", "data": np.array(["three"], dtype="S10")},
    {"fill_value": compound_fill, "data": compound_data},
]


@pytest.fixture(params=fill_values)
def cf_fill_value_hdf5_file(tmp_path: Path, request) -> str:
    filepath = str(tmp_path / "cf_fill_value.nc")

    with h5py.File(filepath, "w") as f:
        dset = f.create_dataset(name="data", data=request.param["data"], chunks=True)
        dim_scale = f.create_dataset(
            name="dim_scale", data=request.param["data"], chunks=True
        )
        dim_scale.make_scale()
        dset.dims[0].attach_scale(dim_scale)
        dset.attrs["_FillValue"] = request.param["fill_value"]

    return filepath


@pytest.fixture
def cf_array_fill_value_hdf5_file(tmp_path: Path) -> str:
    filepath = str(tmp_path / "cf_array_fill_value.nc")

    with h5py.File(filepath, "w") as f:
        data = np.random.random(5)
        dset = f.create_dataset(name="data", data=data, chunks=True)
        dset.attrs["_FillValue"] = np.array([np.nan])

    return filepath


@pytest.fixture()
def chunked_roundtrip_hdf5_s3_file(minio_bucket, cf_array_fill_value_hdf5_file):
    import obstore as obs

    store = obs.store.S3Store(
        minio_bucket["bucket"],
        aws_endpoint=minio_bucket["endpoint"],
        access_key_id=minio_bucket["username"],
        secret_access_key=minio_bucket["password"],
        virtual_hosted_style_request=False,
        client_options={"allow_http": True},
    )
    filepath = "data/cf_array_fill_value.nc"
    obs.put(store, filepath, cf_array_fill_value_hdf5_file)
    return f"s3://{minio_bucket['bucket']}/{filepath}"


@pytest.fixture()
def big_endian_dtype_hdf5_file(tmpdir):
    filepath = f"{tmpdir}/big_endian.nc"
    f = h5py.File(filepath, "w")
    f.create_dataset("data", shape=(10,), dtype=">f4")
    dset = f["data"]
    dset[...] = 10
    return filepath
