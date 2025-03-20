import pytest
from zarr.abc.store import (
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import default_buffer_prototype

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.tests import (
    requires_obstore,
)


@pytest.fixture()
@requires_obstore
def filepath(tmpdir):
    import obstore as obs

    store = obs.store.LocalStore(prefix=tmpdir)
    filepath = "data.tmp"
    obs.put(
        store,
        filepath,
        b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x10\x11\x12\x13\x14\x15\x16",
    )
    return f"{tmpdir}/{filepath}"


@requires_obstore
@pytest.fixture()
def manifest_store(filepath, array_v3_metadata):
    import obstore as obs

    chunk_dict = {
        "0.0": {"path": f"file://{filepath}", "offset": 0, "length": 4},
        "0.1": {"path": f"file://{filepath}", "offset": 4, "length": 4},
        "1.0": {"path": f"file://{filepath}", "offset": 8, "length": 4},
        "1.1": {"path": f"file://{filepath}", "offset": 12, "length": 4},
    }
    manifest = ChunkManifest(entries=chunk_dict)
    chunks = (1, 1)
    shape = (2, 2)
    array_metadata = array_v3_metadata(shape=shape, chunks=chunks)
    manifest_array = ManifestArray(metadata=array_metadata, chunkmanifest=manifest)
    manifest_group = ManifestGroup(
        {"foo": manifest_array, "bar": manifest_array}, attributes={"Zarr": "Hooray!"}
    )
    return ManifestStore(
        stores={"file://": obs.store.LocalStore()}, manifest_group=manifest_group
    )


@requires_obstore
class TestManifestStore:
    def test_manifest_store_properties(self, manifest_store):
        assert manifest_store.read_only
        assert manifest_store.supports_listing
        assert not manifest_store.supports_deletes
        assert not manifest_store.supports_writes

    @pytest.mark.asyncio
    async def test_get(self, manifest_store):
        observed = await manifest_store.get(
            "foo/c/0.0", prototype=default_buffer_prototype()
        )
        assert observed.to_bytes() == b"\x01\x02\x03\x04"
        observed = await manifest_store.get(
            "foo/c/1.0", prototype=default_buffer_prototype()
        )
        assert observed.to_bytes() == b"\x09\x10\x11\x12"
        observed = await manifest_store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=RangeByteRequest(start=1, end=2),
        )
        assert observed.to_bytes() == b"\x02"
        observed = await manifest_store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=OffsetByteRequest(offset=1),
        )
        assert observed.to_bytes() == b"\x02\x03\x04"
        observed = await manifest_store.get(
            "foo/c/0.0",
            prototype=default_buffer_prototype(),
            byte_range=SuffixByteRequest(suffix=2),
        )
        assert observed.to_bytes() == b"\x03\x04"
