import warnings

import h5py  # type: ignore
import numcodecs
import numpy as np

try:
    import imagecodecs  # noqa
except ModuleNotFoundError:
    imagecodecs = None  # type: ignore
    warnings.warn("imagecodecs is required for HDF reader")


from virtualizarr.parsers.hdf.filters import (
    _filter_to_codec,
    cfcodec_from_dataset,
    codecs_from_dataset,
)
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
)


@requires_hdf5plugin
@requires_imagecodecs
class TestFilterToCodec:
    def test_gzip_uses_zlib_numcodec(self):
        codec = _filter_to_codec("gzip", 1)
        assert isinstance(codec, numcodecs.zlib.Zlib)

    def test_lzf(self):
        codec = _filter_to_codec("lzf")
        assert isinstance(codec, imagecodecs.numcodecs.Lzf)

    def test_blosc(self):
        import numcodecs
        from packaging import version

        codec = _filter_to_codec("32001", (2, 2, 8, 800, 9, 2, 1))
        assert isinstance(codec, numcodecs.blosc.Blosc)
        expected_config = {
            "id": "blosc",
            "blocksize": 800,
            "clevel": 9,
            "shuffle": 2,
            "cname": "lz4",
        }
        if (
            version.parse("0.16.1")
            > version.parse(numcodecs.__version__)
            > version.parse("0.15.1")
        ):
            expected_config["typesize"] = None
        assert codec.get_config() == expected_config

    def test_zstd(self):
        codec = _filter_to_codec("32015", (5,))
        assert isinstance(codec, numcodecs.zstd.Zstd)
        config = codec.get_config()
        assert config["id"] == "zstd"
        assert config["level"] == 5

    def test_shuffle(self):
        codec = _filter_to_codec("shuffle", (7,))
        assert isinstance(codec, numcodecs.shuffle.Shuffle)
        expected_config = {"id": "shuffle", "elementsize": 7}
        assert codec.get_config() == expected_config


@requires_hdf5plugin
@requires_imagecodecs
class TestCodecsFromDataSet:
    def test_numcodec_decoding(self, np_uncompressed, filter_encoded_hdf5_file):
        f = h5py.File(filter_encoded_hdf5_file)
        ds = f["data"]
        chunk_info = ds.id.get_chunk_info(0)
        codecs = codecs_from_dataset(ds)
        with open(filter_encoded_hdf5_file, "rb") as file:
            file.seek(chunk_info.byte_offset)
            bytes_read = file.read(chunk_info.size)
            decoded = codecs[0].decode(bytes_read)
            if isinstance(decoded, np.ndarray):
                assert decoded.tobytes() == np_uncompressed.tobytes()
            else:
                assert decoded == np_uncompressed.tobytes()


@requires_hdf5plugin
@requires_imagecodecs
class TestCFCodecFromDataset:
    def test_no_cf_convention(self, filter_encoded_hdf5_file):
        f = h5py.File(filter_encoded_hdf5_file)
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

    def test_cf_add_offset(self, add_offset_hdf5_file):
        f = h5py.File(add_offset_hdf5_file)
        ds = f["data"]
        cf_codec = cfcodec_from_dataset(ds)
        assert cf_codec["target_dtype"] == np.dtype(np.float64)
        assert cf_codec["codec"].scale == 1
        assert cf_codec["codec"].offset == 5
        assert cf_codec["codec"].dtype == "<f8"

    def test_cf_codec_decoding_offset(
        self, add_offset_hdf5_file, np_uncompressed_int16
    ):
        f = h5py.File(add_offset_hdf5_file)
        ds = f["data"]
        chunk_info = ds.id.get_chunk_info(0)
        cfcodec = cfcodec_from_dataset(ds)
        with open(add_offset_hdf5_file, "rb") as file:
            file.seek(chunk_info.byte_offset)
            bytes_read = file.read(chunk_info.size)
            decoded = cfcodec["codec"].decode(bytes_read)
            assert np.array_equal(decoded, np_uncompressed_int16)
            assert decoded.dtype == np.float64

    def test_cf_codec_decoding_scale_offset(
        self, scale_add_offset_hdf5_file, np_uncompressed_int16
    ):
        f = h5py.File(scale_add_offset_hdf5_file)
        ds = f["data"]
        chunk_info = ds.id.get_chunk_info(0)
        cfcodec = cfcodec_from_dataset(ds)
        with open(scale_add_offset_hdf5_file, "rb") as file:
            file.seek(chunk_info.byte_offset)
            bytes_read = file.read(chunk_info.size)
            decoded = cfcodec["codec"].decode(bytes_read)
            assert np.allclose(decoded, np_uncompressed_int16)
            assert decoded.dtype == np.float64
