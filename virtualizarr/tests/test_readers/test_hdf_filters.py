import h5py
import imagecodecs
import numcodecs
import numpy as np

from virtualizarr.readers.hdf_filters import (
    _filter_to_codec,
    cfcodec_from_dataset,
    codecs_from_dataset,
)


class TestFilterToCodec:
    def test_gzip_uses_zlib_numcodec(self):
        codec = _filter_to_codec("gzip", 1)
        assert isinstance(codec, numcodecs.zlib.Zlib)

    def test_lzf(self):
        codec = _filter_to_codec("lzf")
        assert isinstance(codec, imagecodecs.numcodecs.Lzf)

    def test_blosc(self):
        codec = _filter_to_codec("32001", (2, 2, 8, 800, 9, 2, 1))
        assert isinstance(codec, numcodecs.blosc.Blosc)
        expected_config = {
            "id": "blosc",
            "blocksize": 800,
            "clevel": 9,
            "shuffle": 2,
            "cname": "lz4",
        }
        assert codec.get_config() == expected_config


class TestCodecsFromDataSet:
    def test_numcodec_decoding(self, np_uncompressed, filter_encoded_netcdf4_file):
        f = h5py.File(filter_encoded_netcdf4_file)
        ds = f["data"]
        chunk_info = ds.id.get_chunk_info(0)
        codecs = codecs_from_dataset(ds)
        with open(filter_encoded_netcdf4_file, "rb") as file:
            file.seek(chunk_info.byte_offset)
            bytes_read = file.read(chunk_info.size)
            decoded = codecs[0].decode(bytes_read)
            assert decoded == np_uncompressed.tobytes()


class TestCFCodecFromDataset:
    def test_no_cf_convention(self, filter_encoded_netcdf4_file):
        f = h5py.File(filter_encoded_netcdf4_file)
        ds = f["data"]
        cf_codec = cfcodec_from_dataset(ds)
        assert cf_codec is None

    def test_cf_scale_factor(self, netcdf4_file):
        f = h5py.File(netcdf4_file)
        ds = f["air"]
        cf_codec = cfcodec_from_dataset(ds)
        assert cf_codec["target_dtype"] == np.dtype(np.float64)
        assert cf_codec["codec"].scale == 100.0
        assert cf_codec["codec"].offset == 0
        assert cf_codec["codec"].dtype == "<f8"
        assert cf_codec["codec"].astype == "<i2"

    def test_cf_add_offset(self, add_offset_netcdf4_file):
        f = h5py.File(add_offset_netcdf4_file)
        ds = f["data"]
        cf_codec = cfcodec_from_dataset(ds)
        assert cf_codec["target_dtype"] == np.dtype(np.float64)
        assert cf_codec["codec"].scale == 1
        assert cf_codec["codec"].offset == 5
        assert cf_codec["codec"].dtype == "<f8"
