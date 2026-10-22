import h5py
import numpy as np
import pytest
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore

from virtualizarr.parsers import HDFParser
from virtualizarr.parsers.hdf.read_ahead import (
    ChunkIndexReadAhead,
    parse_chunk_index_node,
)
from virtualizarr.tests import requires_hdf5plugin, requires_imagecodecs


class CountingStore:
    """A store that counts the calls made to it, and the ranges asked for in each."""

    def __init__(self):
        self._store = LocalStore()
        self.get_ranges_calls: list[int] = []

    def get(self, *args, **kwargs):
        return self._store.get(*args, **kwargs)

    def head(self, *args, **kwargs):
        return self._store.head(*args, **kwargs)

    def get_ranges(self, path, *, starts, lengths, **kwargs):
        self.get_ranges_calls.append(len(starts))
        return self._store.get_ranges(path, starts=starts, lengths=lengths, **kwargs)


def _write_in_pieces(filepath, *, userblock_size=0):
    """A file whose datasets were written a chunk at a time, so their chunk indexes are spread through it."""
    with h5py.File(filepath, "w", userblock_size=userblock_size) as f:
        one = f.create_dataset("one", shape=(0,), maxshape=(None,), chunks=(4,), dtype="i4")
        two = f.create_dataset(
            "two", shape=(0, 6), maxshape=(None, 6), chunks=(2, 3), dtype="f8"
        )
        for i in range(600):
            one.resize((4 * (i + 1),))
            one[-4:] = i
            two.resize((2 * (i + 1), 6))
            two[-2:] = i
        f.create_dataset("small", data=np.arange(10), chunks=(5,))
        f["contiguous"] = np.arange(5.0)


def _manifests(store):
    return {
        name: (
            array.manifest._paths.tolist(),
            array.manifest._offsets.tolist(),
            array.manifest._lengths.tolist(),
        )
        for name, array in store._group.arrays.items()
    }


@pytest.fixture(params=[0, 512], ids=["no-userblock", "userblock"])
def written_in_pieces(tmp_path, request):
    filepath = tmp_path / "pieces.h5"
    _write_in_pieces(str(filepath), userblock_size=request.param)
    return tmp_path, filepath


@requires_hdf5plugin
@requires_imagecodecs
class TestReadAhead:
    def test_same_result_as_without(self, written_in_pieces):
        _, filepath = written_in_pieces
        url = f"file://{filepath}"
        results = []
        for read_ahead in (True, False):
            registry = ObjectStoreRegistry({"file://": LocalStore()})
            store = HDFParser(read_ahead=read_ahead)(url=url, registry=registry)
            results.append(_manifests(store))
        assert results[0] == results[1]
        assert len(results[0]["one"][1]) == 600

    def test_nodes_are_read_ahead(self, written_in_pieces):
        _, filepath = written_in_pieces
        store = CountingStore()
        path = str(filepath).lstrip("/")
        reader = ChunkIndexReadAhead(
            BlockStoreReader(store, path, block_size=4096), store, path
        )
        with h5py.File(reader, "r") as f:
            for name, n_chunks in (("one", 600), ("two", 1200)):
                info = []
                f[name].id.chunk_iter(info.append)
                assert len(info) == n_chunks
        # The nodes under each root came in calls that each fetched many of them
        assert reader._nodes
        assert max(store.get_ranges_calls) > 1

    def test_reads_of_nodes_come_from_memory(self, written_in_pieces):
        _, filepath = written_in_pieces
        store = CountingStore()
        path = str(filepath).lstrip("/")
        reader = ChunkIndexReadAhead(
            BlockStoreReader(store, path, block_size=4096), store, path
        )
        with h5py.File(reader, "r") as f:
            f["one"].id.chunk_iter(lambda info: None)
            calls = len(store.get_ranges_calls)
            f["one"].id.get_chunk_info(599)
            f["one"].id.chunk_iter(lambda info: None)
        assert len(store.get_ranges_calls) == calls

    def test_off(self, written_in_pieces):
        _, filepath = written_in_pieces
        url = f"file://{filepath}"
        counting = CountingStore()
        registry = ObjectStoreRegistry({"file://": counting})
        store = HDFParser(read_ahead=False)(url=url, registry=registry)
        assert len(store._group.arrays["one"].manifest._offsets) == 600
        # BlockStoreReader asks for one block at a time here; nothing is read ahead
        assert set(counting.get_ranges_calls) == {1}

    def test_without_get_ranges(self, written_in_pieces):
        _, filepath = written_in_pieces

        class OnlyGet:
            def get(self, *args, **kwargs):
                return LocalStore().get(*args, **kwargs)

        url = f"file://{filepath}"
        registry = ObjectStoreRegistry({"file://": OnlyGet()})
        parser = HDFParser(reader_factory=lambda store, path: open("/" + path, "rb"))
        store = parser(url=url, registry=registry)
        assert len(store._group.arrays["one"].manifest._offsets) == 600


def test_parse_chunk_index_node(tmp_path):
    filepath = tmp_path / "pieces.h5"
    _write_in_pieces(str(filepath))
    data = filepath.read_bytes()
    nodes = []
    start = data.find(b"TREE")
    while start >= 0:
        nodes.append(start)
        start = data.find(b"TREE", start + 1)
    parsed = [parse_chunk_index_node(data[n : n + 4096], len(data)) for n in nodes]
    roots = [p for p in parsed if p is not None and p[0] > 0]
    assert roots
    for level, children in roots:
        assert all(data[c : c + 4] == b"TREE" for c in children)


def test_parse_ignores_other_nodes():
    assert parse_chunk_index_node(b"TREE" + bytes([0, 0]) + bytes(100), 1000) is None
    assert parse_chunk_index_node(b"SNOD" + bytes(100), 1000) is None
    assert parse_chunk_index_node(b"TREE", 1000) is None
