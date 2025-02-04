import numpy as np
from zarr.core.metadata.v3 import ArrayV3Metadata

from virtualizarr.zarr import ZArray, zarray_to_v3metadata


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
        compressor={"id": "zlib", "level": 1},
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
