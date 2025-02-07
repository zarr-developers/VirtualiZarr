import numpy as np
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

from virtualizarr.zarr import ZArray, convert_v3_to_v2_metadata, zarray_to_v3metadata


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


def test_zarray_to_v3metadata():
    from zarr.codecs import BytesCodec

    zarray = ZArray(
        shape=(5, 20),
        chunks=(5, 10),
        dtype=np.dtype("int32"),
        fill_value=0,
        order="C",
        compressor={"id": "zlib", "level": 1},
        filters=None,
        zarr_format=3,
    )

    metadata = zarray_to_v3metadata(zarray)

    assert isinstance(metadata, ArrayV3Metadata)
    assert metadata.shape == (5, 20)
    assert metadata.data_type.value == "int32"
    chunk_grid_dict = metadata.chunk_grid.to_dict()
    assert chunk_grid_dict["name"] == "regular"
    assert chunk_grid_dict["configuration"]["chunk_shape"] == (5, 10)
    assert metadata.chunk_key_encoding.name == "default"
    assert metadata.fill_value == np.int32(0)
    assert type(metadata.codecs[0]) is BytesCodec
    metadata_codec_dict = metadata.codecs[1].to_dict()
    assert metadata_codec_dict["name"] == "numcodecs.zlib"
    assert metadata_codec_dict["configuration"]["level"] == 1
    assert metadata.attributes == {}
    assert metadata.dimension_names is None
    assert metadata.storage_transformers == ()


def test_convert_v3_to_v2_metadata(array_v3_metadata):
    shape = (5, 20)
    chunks = (5, 10)
    compressors = [{"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1}]
    filters = [{"id": "delta", "dtype": "<i8"}]

    v3_metadata = array_v3_metadata(shape, chunks, compressors, filters)
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
