import numpy as np
from zarr.codecs import BytesCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.metadata.v2 import ArrayV2Metadata

from virtualizarr.zarr import (
    ZArray,
    convert_to_codec_pipeline,
    convert_v3_to_v2_metadata,
)


def test_convert_to_codec_pipeline():
    expected_default_codecs = BatchedCodecPipeline(
        array_array_codecs=(),
        array_bytes_codec=BytesCodec(endian="little"),
        bytes_bytes_codecs=(),
        batch_size=1,
    )
    # Test with just dtype (default codec pipeline)
    dtype = np.dtype("<i4")
    int_codecs = convert_to_codec_pipeline(dtype=dtype)
    assert int_codecs == expected_default_codecs

    # Test with different dtype
    float_dtype = np.dtype("<f8")
    float_codecs = convert_to_codec_pipeline(dtype=float_dtype)
    assert float_codecs == expected_default_codecs

    # Test with empty codecs list
    empty_codecs = convert_to_codec_pipeline(dtype=dtype, codecs=[])
    assert empty_codecs == expected_default_codecs

    # Test with filters and compressor
    test_codecs = [
        {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}},
        {
            "name": "numcodecs.blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": 1},
        },
    ]

    codecs = convert_to_codec_pipeline(dtype=dtype, codecs=test_codecs)
    assert isinstance(codecs, BatchedCodecPipeline)

    # Verify codec types and order
    array_array_codecs = codecs.array_array_codecs
    assert array_array_codecs[0].codec_name == "numcodecs.delta"
    array_bytes_codec = codecs.array_bytes_codec
    assert isinstance(array_bytes_codec, BytesCodec)
    bytes_bytes_codecs = codecs.bytes_bytes_codecs
    assert bytes_bytes_codecs[0].codec_name == "numcodecs.blosc"
    config = bytes_bytes_codecs[0].codec_config
    assert config["cname"] == "zstd"
    assert config["clevel"] == 5
    assert config["shuffle"] == 1


def test_replace_partial():
    arr = ZArray(shape=(2, 3), chunks=(1, 1), dtype=np.dtype("<i8"))
    result = arr.replace(chunks=(2, 3))
    expected = ZArray(shape=(2, 3), chunks=(2, 3), dtype=np.dtype("<i8"))
    assert result == expected
    assert result.shape == (2, 3)
    assert result.chunks == (2, 3)


def test_replace_total():
    arr = ZArray(shape=(2, 3), chunks=(1, 1), dtype=np.dtype("<i8"))
    kwargs = dict(
        shape=(4, 4),
        chunks=(2, 2),
        dtype=np.dtype("<f8"),
        fill_value=-1.0,
        order="F",
        compressor=[{"id": "zlib", "level": 1}],
        filters=[{"id": "blosc", "clevel": 5}],
        zarr_format=3,
    )
    result = arr.replace(**kwargs)
    expected = ZArray(**kwargs)
    assert result == expected


def test_convert_v3_to_v2_metadata(array_v3_metadata):
    shape = (5, 20)
    chunks = (5, 10)
    codecs = [
        {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}},
        {
            "name": "numcodecs.blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": 1},
        },
    ]

    v3_metadata = array_v3_metadata(shape=shape, chunks=chunks, codecs=codecs)
    v2_metadata = convert_v3_to_v2_metadata(v3_metadata)

    assert isinstance(v2_metadata, ArrayV2Metadata)
    assert v2_metadata.shape == shape
    assert v2_metadata.dtype == np.dtype("int32")
    assert v2_metadata.chunks == chunks
    assert v2_metadata.fill_value == 0
    compressor_config = v2_metadata.compressor.get_config()
    assert compressor_config["id"] == "blosc"
    assert compressor_config["cname"] == "zstd"
    assert compressor_config["clevel"] == 5
    assert compressor_config["shuffle"] == 1
    assert compressor_config["blocksize"] == 0
    filters_config = v2_metadata.filters[0].get_config()
    assert filters_config["id"] == "delta"
    assert filters_config["dtype"] == "<i8"
    assert filters_config["astype"] == "<i8"
    assert v2_metadata.attributes == {}
