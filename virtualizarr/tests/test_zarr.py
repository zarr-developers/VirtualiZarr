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


def test_nan_fill_value_from_kerchunk():
    i_arr = ZArray.from_kerchunk_refs(
        {
            "chunks": [2, 3],
            "compressor": None,
            "dtype": "<i8",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [2, 3],
            "zarr_format": 2,
        }
    )

    assert i_arr.fill_value == 0

    f_arr = ZArray.from_kerchunk_refs(
        {
            "chunks": [2, 3],
            "compressor": None,
            "dtype": "<f8",
            "fill_value": None,
            "filters": None,
            "order": "C",
            "shape": [2, 3],
            "zarr_format": 2,
        }
    )

    assert f_arr.fill_value is np.nan
