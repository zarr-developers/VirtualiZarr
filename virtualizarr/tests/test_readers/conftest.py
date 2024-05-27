import h5py
import hdf5plugin
import numpy as np
import pytest
import xarray as xr
from xarray.tests.test_dataset import create_test_data


@pytest.fixture
def empty_chunks_netcdf4_file(tmpdir):
    ds = xr.Dataset({"data": []})
    filepath = f"{tmpdir}/empty_chunks.nc"
    ds.to_netcdf(filepath, engine="h5netcdf")
    return filepath


@pytest.fixture
def empty_dataset_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/empty_dataset.nc"
    f = h5py.File(filepath, "w")
    f.create_dataset("data", shape=(0,), dtype="f")
    return filepath


@pytest.fixture
def no_chunks_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/no_chunks.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    return filepath


@pytest.fixture
def chunked_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/chunks.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((100, 100))
    f.create_dataset(name="data", data=data, chunks=(50, 50))
    return filepath


@pytest.fixture
def single_dimension_scale_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/single_dimension_scale.nc"
    f = h5py.File(filepath, "w")
    data = [1, 2]
    x = [0, 1]
    f.create_dataset(name="data", data=data)
    f.create_dataset(name="x", data=x)
    f["x"].make_scale()
    f["data"].dims[0].attach_scale(f["x"])
    return filepath


@pytest.fixture
def is_scale_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/is_scale.nc"
    f = h5py.File(filepath, "w")
    data = [1, 2]
    f.create_dataset(name="data", data=data)
    f["data"].make_scale()
    return filepath


@pytest.fixture
def multiple_dimension_scales_netcdf4_file(tmpdir):
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
    return filepath


@pytest.fixture
def chunked_dimensions_netcdf4_file(tmpdir):
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
    return filepath


@pytest.fixture
def string_attributes_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/attributes.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    f["data"].attrs["attribute_name"] = "attribute_name"
    f["data"].attrs["attribute_name2"] = "attribute_name2"
    return filepath


@pytest.fixture
def root_attributes_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/root_attributes.nc"
    f = h5py.File(filepath, "w")
    f.attrs["attribute_name"] = "attribute_name"
    return filepath


@pytest.fixture
def group_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/group.nc"
    f = h5py.File(filepath, "w")
    f.create_group("group")
    return filepath


@pytest.fixture
def multiple_datasets_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/multiple_datasets.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    f.create_dataset(name="data2", data=data, chunks=None)
    return filepath


@pytest.fixture
def np_uncompressed():
    return np.arange(100)


@pytest.fixture(params=["gzip", "blosc_lz4"])
def filter_encoded_netcdf4_file(tmpdir, np_uncompressed, request):
    filepath = f"{tmpdir}/{request.param}.nc"
    f = h5py.File(filepath, "w")
    if request.param == "gzip":
        f.create_dataset(
            name="data", data=np_uncompressed, compression="gzip", compression_opts=1
        )
    if request.param == "blosc_lz4":
        f.create_dataset(
            name="data",
            data=np_uncompressed,
            **hdf5plugin.Blosc(cname="lz4", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE),
        )
    return filepath


@pytest.fixture(params=["gzip"])
def filter_encoded_xarray_h5netcdf_file(tmpdir, request):
    ds = xr.tutorial.open_dataset("air_temperature")
    encoding = {}
    if request.param == "gzip":
        encoding_config = {"zlib": True, "complevel": 1}

    for var_name in ds.variables:
        encoding[var_name] = encoding_config

    filepath = f"{tmpdir}/{request.param}_xarray.nc"
    ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
    return filepath


@pytest.fixture(params=["blosc_zlib"])
def filter_encoded_xarray_netcdf4_file(tmpdir, request):
    ds = create_test_data(dim_sizes=(20, 80, 10))
    if "blosc" in request.param:
        encoding_config = {
            "compression": request.param,
            "chunksizes": (20, 40),
            "original_shape": ds.var2.shape,
            "blosc_shuffle": 1,
            "fletcher32": False,
        }

    ds["var2"].encoding.update(encoding_config)
    filepath = f"{tmpdir}/{request.param}_xarray.nc"
    ds.to_netcdf(filepath, engine="netcdf4")
    return filepath


@pytest.fixture
def add_offset_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/offset.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    f["data"].attrs.create(name="add_offset", data=5)
    return filepath
