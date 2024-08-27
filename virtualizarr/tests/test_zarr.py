import numpy as np

from virtualizarr.zarr import ZArray


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
