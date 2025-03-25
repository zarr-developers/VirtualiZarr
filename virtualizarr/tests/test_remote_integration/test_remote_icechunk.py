import time
from typing import TYPE_CHECKING, Optional

import pytest

pytest.importorskip("icechunk")

import numpy as np
import xarray as xr
import zarr
from zarr.core.metadata import ArrayV3Metadata

if TYPE_CHECKING:
    from icechunk import (  # type: ignore[import-not-found]
        IcechunkStore,
        Storage,
    )


@pytest.fixture(scope="function")
def icechunk_miniostore(minio_bucket) -> "Storage":
    # Based on https://github.com/earth-mover/icechunk/blob/3374ca4968e0989b78643f57c8dda1fee0e8da2e/icechunk-python/tests/test_gc.py
    import icechunk as ic

    prefix = "test-repo__" + str(time.time())

    repo = ic.Repository.create(
        storage=ic.s3_storage(
            endpoint_url="http://localhost:9000",
            allow_http=True,
            force_path_style=True,
            region="us-east-1",
            bucket=minio_bucket["bucket"],
            prefix=prefix,
            access_key_id=minio_bucket["username"],
            secret_access_key=minio_bucket["password"],
        ),
        config=ic.RepositoryConfig(inline_chunk_threshold_bytes=0),
    )
    session = repo.writable_session("main")
    return session.store


@pytest.mark.parametrize("group_path", [None, "", "/a", "a", "/a/b", "a/b", "a/b/"])
def test_write_new_virtual_variable(
    icechunk_miniostore: "IcechunkStore",
    vds_with_manifest_arrays: xr.Dataset,
    group_path: Optional[str],
):
    vds = vds_with_manifest_arrays

    vds.virtualize.to_icechunk(icechunk_miniostore, group=group_path)

    # check attrs
    group = zarr.group(store=icechunk_miniostore, path=group_path)
    assert isinstance(group, zarr.Group)
    assert group.attrs.asdict() == {"something": 0}

    # TODO check against vds, then perhaps parametrize?

    # check array exists
    assert "a" in group
    arr = group["a"]
    assert isinstance(arr, zarr.Array)

    # check array metadata
    assert arr.metadata.zarr_format == 3
    assert arr.shape == (2, 3)
    assert arr.chunks == (2, 3)
    assert arr.dtype == np.dtype("<i8")
    assert arr.order == "C"
    assert arr.fill_value == 0
    # TODO check compressor, filters?
    #

    # check array attrs
    # TODO somehow this is broken by setting the dimension names???
    # assert dict(arr.attrs) == {"units": "km"}

    # check dimensions
    if isinstance(arr.metadata, ArrayV3Metadata):
        assert arr.metadata.dimension_names == ("x", "y")
