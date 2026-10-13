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


def test_copy_and_replace_metadata_rejects_non_axis_change(array_v3_metadata):
    """
    Guard on an internal invariant: an outer chunk shape can only change by adding or
    removing length-1 axes, so a sharding codec's inner chunk shape can always be
    realigned onto the new axes. No array operation can currently violate this - this
    tests the defensive branch directly, so a future caller gets a clear error rather
    than a silently mismatched shard config.
    """
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
