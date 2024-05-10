import h5py
import numpy as np
import pytest
import xarray as xr


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
def string_attribute_netcdf4_file(tmpdir):
    filepath = f"{tmpdir}/attributes.nc"
    f = h5py.File(filepath, "w")
    data = np.random.random((10, 10))
    f.create_dataset(name="data", data=data, chunks=None)
    f["data"].attrs["attribute_name"] = "attribute_name"
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


@pytest.fixture
def gzip_filter_netcdf4_file(tmpdir, np_uncompressed):
    filepath = f"{tmpdir}/gzip.nc"
    f = h5py.File(filepath, "w")
    f.create_dataset(name="data", data=np_uncompressed, compression="gzip", compression_opts=1)
    return filepath


@pytest.fixture
def gzip_filter_xarray_netcdf4_file(tmpdir):
    ds = xr.tutorial.open_dataset("air_temperature")
    encoding = {}
    for var_name in ds.variables:
        #  encoding[var_name] = {"zlib": True, "compression_opts": 1}
        encoding[var_name] = {"compression": "gzip", "compression_opts": 1}

    filepath = f"{tmpdir}/gzip_xarray.nc"
    ds.to_netcdf(filepath, engine="h5netcdf", encoding=encoding)
    return filepath
