import numpy as np
import pytest
import xarray as xr

from conftest import sharding_codec as _sharding_codec
from virtualizarr.manifests.utils import copy_and_replace_metadata


@pytest.fixture
def dataset() -> xr.Dataset:
    return xr.Dataset(
        {"x": xr.DataArray([10, 20, 30], dims="a", coords={"a": [0, 1, 2]})}
    )


def test_copy_and_replace_metadata(array_v3_metadata):
    old_metadata = array_v3_metadata(
        shape=(10, 10),
        data_type=np.dtype("float32"),
        chunks=(5, 5),
        fill_value=0,
    )

    new_shape = (20, 20)
    new_chunks = (10, 10)

    # Test updating both shape and chunk shape
    updated_metadata = copy_and_replace_metadata(
        old_metadata, new_shape=new_shape, new_chunks=new_chunks
    )
    assert updated_metadata.shape == tuple(new_shape)
    assert updated_metadata.chunks == tuple(new_chunks)
    # Test other values are still the same
    assert updated_metadata.data_type == old_metadata.data_type
    assert updated_metadata.fill_value == old_metadata.fill_value


class TestCopyAndReplaceMetadataSharding:
    """
    A sharding codec's inner chunk_shape must have the same ndim as the array, so adding
    length-1 axes to the outer chunk shape has to add them to the shard config too.
    """

    def test_prepended_axis(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((45, 45))],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(1, 90, 180), new_chunks=(1, 90, 180)
        )

        assert updated_metadata.shards == (1, 90, 180)
        assert updated_metadata.chunks == (1, 45, 45)

    def test_axis_inserted_in_middle(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((45, 45))],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(90, 1, 180), new_chunks=(90, 1, 180)
        )

        assert updated_metadata.chunks == (45, 1, 45)

    def test_multiple_prepended_axes(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((45, 45))],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(1, 1, 90, 180), new_chunks=(1, 1, 90, 180)
        )

        assert updated_metadata.chunks == (1, 1, 45, 45)

    def test_nested_shards(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
            codecs=[
                _sharding_codec((45, 90), inner_codecs=[_sharding_codec((45, 45))])
            ],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(1, 90, 180), new_chunks=(1, 90, 180)
        )

        (outer_shard,) = updated_metadata.codecs
        assert outer_shard.chunk_shape == (1, 45, 90)
        (inner_shard,) = outer_shard.codecs
        assert inner_shard.chunk_shape == (1, 45, 45)

    def test_ambiguous_alignment_when_old_chunks_contain_ones(self, array_v3_metadata):
        # (1, 45) could align with either the first or second axis of (1, 1, 45), but
        # both readings insert a 1 next to an existing 1, so the result is the same
        old_metadata = array_v3_metadata(
            shape=(1, 180),
            chunks=(1, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((1, 45))],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(1, 1, 180), new_chunks=(1, 1, 180)
        )

        assert updated_metadata.chunks == (1, 1, 45)

    def test_dropped_axis(self, array_v3_metadata):
        # integer indexing drops an axis, which is only legal where the chunk length is
        # already 1 - so the inner chunk shape holds a 1 there and can drop it too
        old_metadata = array_v3_metadata(
            shape=(4, 90, 180),
            chunks=(1, 90, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((1, 45, 45))],
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(90, 180), new_chunks=(90, 180)
        )

        assert updated_metadata.shards == (90, 180)
        assert updated_metadata.chunks == (45, 45)

    def test_unsharded_codecs_untouched(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
        )

        updated_metadata = copy_and_replace_metadata(
            old_metadata, new_shape=(1, 90, 180), new_chunks=(1, 90, 180)
        )

        assert [codec.to_dict() for codec in updated_metadata.codecs] == [
            codec.to_dict() for codec in old_metadata.codecs
        ]

    def test_rejects_chunk_shape_that_is_not_an_axis_change(self, array_v3_metadata):
        old_metadata = array_v3_metadata(
            shape=(90, 180),
            chunks=(90, 180),
            data_type=np.dtype("float32"),
            codecs=[_sharding_codec((45, 45))],
        )

        with pytest.raises(ValueError, match="length-1 axes added or removed"):
            copy_and_replace_metadata(
                old_metadata, new_shape=(2, 90, 180), new_chunks=(2, 90, 180)
            )
