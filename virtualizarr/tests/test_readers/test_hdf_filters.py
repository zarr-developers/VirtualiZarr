import h5py
import numcodecs
import pytest

from virtualizarr.readers.hdf_filters import (
    _filter_to_codec,
    codecs_from_dataset,
)


class TestFilterToCodec:
    def test_gzip_uses_zlib_nomcodec(self):
        codec = _filter_to_codec("gzip", 1)
        assert isinstance(codec, numcodecs.zlib.Zlib)

    def test_lzf_not_available(self):
        with pytest.raises(ValueError, match="codec not available"):
            _filter_to_codec("lzf")


class TestCodecsFromDataSet:
    def test_gzip(self, np_uncompressed, gzip_filter_netcdf4_file):
        f = h5py.File(gzip_filter_netcdf4_file)
        ds = f["data"]
        chunk_info = ds.id.get_chunk_info(0)
        codecs = codecs_from_dataset(ds)
        with open(gzip_filter_netcdf4_file, 'rb') as file:
            file.seek(chunk_info.byte_offset)
            bytes_read = file.read(chunk_info.size)
            decoded = codecs[0].decode(bytes_read)
            assert decoded == np_uncompressed.tobytes()
