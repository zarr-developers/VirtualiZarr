import numpy as np
from zarr.core.metadata.v2 import ArrayV2Metadata

from virtualizarr.zarr import ZArray, convert_v3_to_v2_metadata


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
