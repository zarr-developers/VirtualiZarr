import json
import pickle

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
)
from virtualizarr.tests import requires_minio, requires_obstore


@pytest.fixture()
def local_store(tmpdir, array_v3_metadata):
    import obstore as obs

    store = obs.store.LocalStore(prefix=tmpdir)
    filepath = "data.tmp"
    obs.put(
        store,
        filepath,
        b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15\x16",
    )
    chunk_dict = {
        "0.0": {"path": f"file://{tmpdir}/{filepath}", "offset": 0, "length": 4},
        "0.1": {"path": f"file://{tmpdir}/{filepath}", "offset": 4, "length": 4},
        "1.0": {"path": f"file://{tmpdir}/{filepath}", "offset": 8, "length": 4},
        "1.1": {"path": f"file://{tmpdir}/{filepath}", "offset": 12, "length": 4},
    }
    manifest = ChunkManifest(entries=chunk_dict)
    chunks = (1, 4)
    shape = (2, 8)
    array_metadata = array_v3_metadata(shape=shape, chunks=chunks)
    manifest_array = ManifestArray(metadata=array_metadata, chunkmanifest=manifest)
    manifest_group = ManifestGroup(
        {"foo": manifest_array, "bar": manifest_array}, attributes={"Zarr": "Hooray!"}
    )
    return ManifestStore(
        stores={"file://": obs.store.LocalStore()}, manifest_group=manifest_group
    )


@pytest.fixture()
def s3_store(minio_bucket, array_v3_metadata):
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
    obs.put(
        store,
        filepath,
        b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15\x16",
    )
    chunk_dict = {
        "0.0": {
            "path": f"s3://{minio_bucket['bucket']}/{filepath}",
            "offset": 0,
            "length": 4,
        },
        "0.1": {
            "path": f"s3://{minio_bucket['bucket']}/{filepath}",
            "offset": 4,
            "length": 4,
        },
        "1.0": {
            "path": f"s3://{minio_bucket['bucket']}/{filepath}",
            "offset": 8,
            "length": 4,
        },
        "1.1": {
            "path": f"s3://{minio_bucket['bucket']}/{filepath}",
            "offset": 12,
            "length": 4,
        },
    }
    manifest = ChunkManifest(entries=chunk_dict)
    chunks = (1, 4)
    shape = (2, 8)
    array_metadata = array_v3_metadata(shape=shape, chunks=chunks)
    manifest_array = ManifestArray(metadata=array_metadata, chunkmanifest=manifest)
    manifest_group = ManifestGroup(
        {"foo": manifest_array, "bar": manifest_array}, attributes={"Zarr": "Hooray!"}
    )
    return ManifestStore(stores={"s3://": store}, manifest_group=manifest_group)


@requires_obstore
class TestManifestStore:
    def test_manifest_store_properties(self, local_store):
        assert local_store.read_only
        assert local_store.supports_listing
        assert not local_store.supports_deletes
        assert not local_store.supports_writes
        assert not local_store.supports_partial_writes

    @requires_minio
    @pytest.mark.asyncio
    @pytest.mark.parametrize("manifest_store", ["local_store", "s3_store"])
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

    @requires_minio
    @pytest.mark.asyncio
    @pytest.mark.parametrize("manifest_store", ["local_store", "s3_store"])
    async def test_get_metadata(self, manifest_store, request):
        store = request.getfixturevalue(manifest_store)
        observed = await store.get(
            "foo/zarr.json", prototype=default_buffer_prototype()
        )
        metadata = json.loads(observed.to_bytes())
        assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [1, 4]
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

    @requires_minio
    @pytest.mark.asyncio
    @pytest.mark.parametrize("manifest_store", ["local_store", "s3_store"])
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
