from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
from zarr.abc.store import (
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import _collect_aiterator

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
    ObjectStoreRegistry,
)
from virtualizarr.manifests.store import default_object_store
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.tests import (
    requires_hdf5plugin,
    requires_imagecodecs,
    requires_minio,
    requires_network,
    requires_obstore,
)

if TYPE_CHECKING:
    from obstore.store import ObjectStore


def _generate_manifest_store(
    store: ObjectStore, *, prefix: str, filepath: str
) -> ManifestStore:
    """
    Generate a ManifestStore for testing.

    This puts a sequence of 16 bytes in a file, which can simulate storing
    4 contiguous uncompressed 4-byte chunks (or 8 2-byte chunks, etc). This
    provides an easily understandable structure for testing ManifestStore's
    ability to redirect Zarr chunk key requests and extract subsets of the file.

    Parameters
    ----------
    store
        ObjectStore instance for holding the file
    prefix
        Prefix to use to identify the ObjectStore in the ManifestStore
    filepath
        Filepath for storing temporary testing file

    Returns
    -------
    ManifestStore
    """
    import obstore as obs

    obs.put(
        store,
        filepath,
        b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15\x16",
    )
    chunk_dict = {
        "0.0": {"path": f"{prefix}/{filepath}", "offset": 0, "length": 4},
        "0.1": {"path": f"{prefix}/{filepath}", "offset": 4, "length": 4},
        "1.0": {"path": f"{prefix}/{filepath}", "offset": 8, "length": 4},
        "1.1": {"path": f"{prefix}/{filepath}", "offset": 12, "length": 4},
    }
    manifest = ChunkManifest(entries=chunk_dict)
    codecs = [{"configuration": {"endian": "little"}, "name": "bytes"}]
    array_metadata = create_v3_array_metadata(
        shape=(4, 4),
        chunk_shape=(2, 2),
        data_type=np.dtype("int32"),
        codecs=codecs,
        chunk_key_encoding={"name": "default", "separator": "."},
        fill_value=0,
    )
    manifest_array = ManifestArray(metadata=array_metadata, chunkmanifest=manifest)
    manifest_group = ManifestGroup(
        arrays={"foo": manifest_array, "bar": manifest_array},
        attributes={"Zarr": "Hooray!"},
    )
    registry = ObjectStoreRegistry({prefix: store})
    return ManifestStore(store_registry=registry, group=manifest_group)


@pytest.fixture()
def local_store(tmpdir):
    import obstore as obs

    store = obs.store.LocalStore()
    filepath = f"{tmpdir}/data.tmp"
    prefix = "file://"
    return _generate_manifest_store(
        store=store,
        prefix=prefix,
        filepath=filepath,
    )


@pytest.fixture()
def s3_store(minio_bucket):
    import obstore as obs

    store = obs.store.S3Store(
        minio_bucket["bucket"],
        aws_endpoint=minio_bucket["endpoint"],
        access_key_id=minio_bucket["username"],
        secret_access_key=minio_bucket["password"],
        virtual_hosted_style_request=False,
        client_options={"allow_http": True},
    )
    filepath = "data.tmp"
    prefix = f"s3://{minio_bucket['bucket']}"
    return _generate_manifest_store(
        store=store,
        prefix=prefix,
        filepath=filepath,
    )


@requires_obstore
@requires_minio
def test_default_object_store_s3(minio_bucket):
    from obstore.store import S3Store

    filepath = f"s3://{minio_bucket['bucket']}/data/data.tmp"
    store = default_object_store(
        filepath,
    )
    assert isinstance(store, S3Store)


@requires_obstore
@requires_minio
def test_default_object_store_http(minio_bucket):
    from obstore.store import HTTPStore

    filepath = minio_bucket["endpoint"]
    store = default_object_store(
        filepath,
    )
    assert isinstance(store, HTTPStore)


@requires_obstore
def test_default_object_store_local(tmpdir):
    from obstore.store import LocalStore

    filepath = f"{tmpdir}/data.tmp"
    store = default_object_store(filepath)
    assert isinstance(store, LocalStore)


@requires_network
@requires_obstore
def test_default_region_raises():
    file = "s3://cworthy/oae-efficiency-atlas/data/experiments/000/01/alk-forcing.000-1999-01.pop.h.0347-01.nc"
    with pytest.raises(
        ValueError, match="Unable to automatically determine region for bucket*"
    ):
        default_object_store(file)


@requires_obstore
class TestManifestStore:
    def test_manifest_store_properties(self, local_store):
        assert local_store.read_only
        assert local_store.supports_listing
        assert not local_store.supports_deletes
        assert not local_store.supports_writes
        assert not local_store.supports_partial_writes

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest_store",
        ["local_store", pytest.param("s3_store", marks=requires_minio)],
    )
    async def test_get_data(self, manifest_store, request):
        store = request.getfixturevalue(manifest_store)
        observed = await store.get("foo/c/0.0", prototype=default_buffer_prototype())
        assert observed.to_bytes() == b"\x01\x02\x03\x04"
        observed = await store.get("foo/c/1.0", prototype=default_buffer_prototype())
        assert observed.to_bytes() == b"\x09\x10\x11\x12"
        observed = await store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=RangeByteRequest(start=1, end=2),
        )
        assert observed.to_bytes() == b"\x02"
        observed = await store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=OffsetByteRequest(offset=1),
        )
        assert observed.to_bytes() == b"\x02\x03\x04"
        observed = await store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=SuffixByteRequest(suffix=2),
        )
        assert observed.to_bytes() == b"\x03\x04"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest_store",
        ["local_store", pytest.param("s3_store", marks=requires_minio)],
    )
    async def test_get_metadata(self, manifest_store, request):
        store = request.getfixturevalue(manifest_store)
        observed = await store.get(
            "foo/zarr.json", prototype=default_buffer_prototype()
        )
        metadata = json.loads(observed.to_bytes())
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [2, 2]
        assert metadata["node_type"] == "array"
        assert metadata["zarr_format"] == 3

        observed = await store.get("zarr.json", prototype=default_buffer_prototype())
        metadata = json.loads(observed.to_bytes())
        assert metadata["node_type"] == "group"
        assert metadata["zarr_format"] == 3
        assert metadata["attributes"]["Zarr"] == "Hooray!"

    @pytest.mark.asyncio
    async def test_pickling(self, local_store):
        new_store = pickle.loads(pickle.dumps(local_store))
        assert isinstance(new_store, ManifestStore)
        # Check new store works
        observed = await local_store.get(
            "foo/c/0.0", prototype=default_buffer_prototype()
        )
        assert observed.to_bytes() == b"\x01\x02\x03\x04"
        # Check old store works
        observed = await new_store.get(
            "foo/c/0.0", prototype=default_buffer_prototype()
        )
        assert observed.to_bytes() == b"\x01\x02\x03\x04"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest_store",
        ["local_store", pytest.param("s3_store", marks=requires_minio)],
    )
    async def test_list_dir(self, manifest_store, request) -> None:
        store = request.getfixturevalue(manifest_store)
        observed = await _collect_aiterator(store.list_dir(""))
        assert observed == ("zarr.json", "foo", "bar")

    @pytest.mark.asyncio
    async def test_store_raises(self, local_store) -> None:
        with pytest.raises(NotImplementedError):
            await local_store.set("foo/zarr.json", 1)
        with pytest.raises(NotImplementedError):
            await local_store.set_if_not_exists("foo/zarr.json", 1)
        with pytest.raises(NotImplementedError):
            await local_store.delete("foo")


@requires_obstore
@requires_hdf5plugin
@requires_imagecodecs
class TestToVirtualXarray:
    @pytest.mark.parametrize(
        "loadable_variables, expected_loadable_variables",
        [
            ([], []),
            (["t"], ["t"]),
            (["T", "t"], ["T", "t"]),
            (["T", "elevation"], ["T", "elevation"]),
            (None, ["t"]),
        ],
    )
    def test_single_group_to_dataset(
        self, manifest_array, loadable_variables, expected_loadable_variables
    ):
        marr1 = manifest_array(
            shape=(3, 2, 5), chunks=(1, 2, 1), dimension_names=["x", "y", "t"]
        )
        marr2 = manifest_array(shape=(3, 2), chunks=(1, 2), dimension_names=["x", "y"])
        marr3 = manifest_array(shape=(5,), chunks=(5,), dimension_names=["t"])

        manifest_group = ManifestGroup(
            arrays={
                "T": marr1,  # data variable
                "elevation": marr2,  # 2D coordinate
                "t": marr3,  # 1D dimension coordinate
            },
            attributes={"coordinates": "elevation t", "ham": "eggs"},
        )

        manifest_store = ManifestStore(manifest_group)

        vds = manifest_store.to_virtual_dataset(loadable_variables=loadable_variables)
        assert set(vds.variables) == set(["T", "elevation", "t"])
        assert vds.attrs == {"ham": "eggs"}
        assert set(vds.dims) == set(["x", "y", "t"])
        assert set(vds.coords) == set(["elevation", "t"])

        var_name = "T"
        var = vds.variables[var_name]
        assert set(var.dims) == set(["x", "y", "t"])
        if var_name in expected_loadable_variables:
            assert isinstance(var.data, np.ndarray)
        else:
            assert isinstance(var.data, ManifestArray)
            # check dims info is not duplicated in two places
            assert var.data.metadata.dimension_names is None
            assert var.attrs == {}
