from __future__ import annotations

import numpy as np
import pytest
from obspec_utils.registry import ObjectStoreRegistry
from zarr.core.buffer import default_buffer_prototype

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import _coalesce_members, _Member
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.tests import requires_obstore

# 16 distinct bytes, enough for four contiguous 4-byte chunks.
FILE_BYTES = bytes(range(1, 17))
_CODECS = [{"configuration": {"endian": "little"}, "name": "bytes"}]


def _build_store(
    tmpdir,
    *,
    files: dict[str, bytes] | None = None,
    entries: dict[str, dict] | None = None,
    coalesce_max_gap_bytes: int = 1024 * 1024,
    coalesce_max_bytes: int = 8 * 1024 * 1024,
) -> ManifestStore:
    """Build a ManifestStore over a LocalStore with one array ``foo`` (4x4, 2x2 chunks)."""
    import obstore as obs

    store = obs.store.LocalStore()
    prefix = "file://"
    files = files or {f"{tmpdir}/data.tmp": FILE_BYTES}
    for filepath, data in files.items():
        obs.put(store, filepath, data)

    if entries is None:
        fp = next(iter(files))
        entries = {
            "0.0": {"path": f"{prefix}/{fp}", "offset": 0, "length": 4},
            "0.1": {"path": f"{prefix}/{fp}", "offset": 4, "length": 4},
            "1.0": {"path": f"{prefix}/{fp}", "offset": 8, "length": 4},
            "1.1": {"path": f"{prefix}/{fp}", "offset": 12, "length": 4},
        }
    manifest = ChunkManifest(entries=entries)
    metadata = create_v3_array_metadata(
        shape=(4, 4),
        chunk_shape=(2, 2),
        data_type=np.dtype("int32"),
        codecs=_CODECS,
        chunk_key_encoding={"name": "default", "separator": "."},
        fill_value=0,
    )
    marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    group = ManifestGroup(arrays={"foo": marr})
    registry = ObjectStoreRegistry({prefix: store})
    return ManifestStore(
        group=group,
        registry=registry,
        coalesce_max_gap_bytes=coalesce_max_gap_bytes,
        coalesce_max_bytes=coalesce_max_bytes,
    )


async def _collect(store: ManifestStore, requests) -> dict[int, object]:
    """Drain get_many into an index -> buffer mapping, asserting each index appears once."""
    collected: dict[int, object] = {}
    async for batch in store.get_many(requests, prototype=default_buffer_prototype()):
        for index, buffer in batch:
            assert index not in collected, "each request must be reported exactly once"
            collected[index] = buffer
    return collected


class _RecordingStore:
    """Proxy that records each ranged read (``get_range_async``) and forwards
    everything else.

    Installed by monkeypatching ``ManifestStore._resolve_store_and_path`` so we
    can see exactly which byte ranges get_many issues after coalescing.
    """

    def __init__(self, inner, log: list[dict]) -> None:
        self._inner = inner
        self._log = log

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_range_async(self, path, *, start, end):
        self._log.append({"path": path, "start": start, "end": end})
        return await self._inner.get_range_async(path, start=start, end=end)


@pytest.fixture()
def record_range_calls(monkeypatch):
    """Record every get_ranges_async call ManifestStore.get_many issues."""
    log: list[dict] = []
    original = ManifestStore._resolve_store_and_path
    proxies: dict[int, _RecordingStore] = {}

    def patched(self, path):
        inner, path_in_store = original(self, path)
        proxy = proxies.get(id(inner))
        if proxy is None:
            proxy = _RecordingStore(inner, log)
            proxies[id(inner)] = proxy
        return proxy, path_in_store

    monkeypatch.setattr(ManifestStore, "_resolve_store_and_path", patched)
    return log


@requires_obstore
class TestGetMany:
    pytestmark = pytest.mark.asyncio

    async def test_matches_individual_get(self, tmpdir):
        store = _build_store(tmpdir)
        keys = ["foo/c.0.0", "foo/c.0.1", "foo/c.1.0", "foo/c.1.1"]
        collected = await _collect(store, keys)

        assert set(collected) == set(range(len(keys)))
        for index, key in enumerate(keys):
            expected = await store.get(key, prototype=default_buffer_prototype())
            assert collected[index].to_bytes() == expected.to_bytes()
        # sanity: the four chunks are the four consecutive 4-byte slices
        assert collected[0].to_bytes() == FILE_BYTES[0:4]
        assert collected[3].to_bytes() == FILE_BYTES[12:16]

    async def test_accepts_key_range_tuples(self, tmpdir):
        from zarr.abc.store import RangeByteRequest

        store = _build_store(tmpdir)
        requests = [
            "foo/c.0.0",
            ("foo/c.0.1", None),
            ("foo/c.1.0", RangeByteRequest(1, 3)),  # middle two bytes of that chunk
        ]
        collected = await _collect(store, requests)
        assert collected[0].to_bytes() == FILE_BYTES[0:4]
        assert collected[1].to_bytes() == FILE_BYTES[4:8]
        assert collected[2].to_bytes() == FILE_BYTES[8:12][1:3]

    async def test_missing_chunk_is_none(self, tmpdir):
        prefix = "file://"
        fp = f"{tmpdir}/data.tmp"
        # omit chunk "1.1" so it has no manifest entry
        entries = {
            "0.0": {"path": f"{prefix}/{fp}", "offset": 0, "length": 4},
            "0.1": {"path": f"{prefix}/{fp}", "offset": 4, "length": 4},
            "1.0": {"path": f"{prefix}/{fp}", "offset": 8, "length": 4},
        }
        store = _build_store(tmpdir, entries=entries)
        collected = await _collect(store, ["foo/c.0.0", "foo/c.1.1"])
        assert collected[0].to_bytes() == FILE_BYTES[0:4]
        assert collected[1] is None  # missing -> None, not omitted

    async def test_metadata_key_served_individually(self, tmpdir):
        store = _build_store(tmpdir)
        collected = await _collect(store, ["foo/zarr.json", "foo/c.0.0"])
        assert collected[0] is not None  # zarr.json metadata document
        assert b'"zarr_format"' in collected[0].to_bytes()
        assert collected[1].to_bytes() == FILE_BYTES[0:4]

    async def test_empty_requests(self, tmpdir):
        store = _build_store(tmpdir)
        collected = await _collect(store, [])
        assert collected == {}

    async def test_coalesces_contiguous_into_one_read(self, tmpdir, record_range_calls):
        store = _build_store(tmpdir)
        keys = ["foo/c.0.0", "foo/c.0.1", "foo/c.1.0", "foo/c.1.1"]
        collected = await _collect(store, keys)

        # the four contiguous chunks are served by a single ranged read [0, 16)
        assert [(c["start"], c["end"]) for c in record_range_calls] == [(0, 16)]
        # and the bytes were sliced back to the right chunks
        assert [collected[i].to_bytes() for i in range(4)] == [
            FILE_BYTES[0:4],
            FILE_BYTES[4:8],
            FILE_BYTES[8:12],
            FILE_BYTES[12:16],
        ]

    async def test_groups_requests_by_file(self, tmpdir, record_range_calls):
        prefix = "file://"
        fp_a = f"{tmpdir}/a.tmp"
        fp_b = f"{tmpdir}/b.tmp"
        entries = {
            "0.0": {"path": f"{prefix}/{fp_a}", "offset": 0, "length": 4},
            "0.1": {"path": f"{prefix}/{fp_a}", "offset": 4, "length": 4},
            "1.0": {"path": f"{prefix}/{fp_b}", "offset": 0, "length": 4},
            "1.1": {"path": f"{prefix}/{fp_b}", "offset": 4, "length": 4},
        }
        store = _build_store(
            tmpdir, files={fp_a: FILE_BYTES, fp_b: FILE_BYTES}, entries=entries
        )
        await _collect(store, ["foo/c.0.0", "foo/c.0.1", "foo/c.1.0", "foo/c.1.1"])

        # one read per distinct source file, each spanning that file's two chunks
        assert len(record_range_calls) == 2
        for call in record_range_calls:
            assert (call["start"], call["end"]) == (0, 8)

    async def test_gap_larger_than_max_gap_is_not_merged(
        self, tmpdir, record_range_calls
    ):
        prefix = "file://"
        fp = f"{tmpdir}/data.tmp"
        # two chunks 996 bytes apart in the file
        entries = {
            "0.0": {"path": f"{prefix}/{fp}", "offset": 0, "length": 4},
            "0.1": {"path": f"{prefix}/{fp}", "offset": 1000, "length": 4},
        }
        store = _build_store(
            tmpdir,
            files={fp: bytes(2048)},
            entries=entries,
            coalesce_max_gap_bytes=8,  # < 996 gap -> keep separate
        )
        await _collect(store, ["foo/c.0.0", "foo/c.0.1"])
        assert sorted((c["start"], c["end"]) for c in record_range_calls) == [
            (0, 4),
            (1000, 1004),
        ]

    async def test_max_bytes_caps_a_merge(self, tmpdir, record_range_calls):
        prefix = "file://"
        fp = f"{tmpdir}/data.tmp"
        entries = {
            "0.0": {"path": f"{prefix}/{fp}", "offset": 0, "length": 4},
            "0.1": {"path": f"{prefix}/{fp}", "offset": 500, "length": 4},
        }
        # the 500-byte gap is within max_gap, but merging would make a 504-byte
        # read which exceeds max_bytes, so the two chunks stay separate.
        store = _build_store(
            tmpdir,
            files={fp: bytes(2048)},
            entries=entries,
            coalesce_max_gap_bytes=1000,
            coalesce_max_bytes=100,
        )
        await _collect(store, ["foo/c.0.0", "foo/c.0.1"])
        assert sorted((c["start"], c["end"]) for c in record_range_calls) == [
            (0, 4),
            (500, 504),
        ]


@requires_obstore
class TestCoalesceMembers:
    def _members(self, *ranges):
        return [_Member(i, start, end) for i, (start, end) in enumerate(ranges)]

    def test_merges_contiguous_and_near(self):
        # 0-4 and 4-8 touch; 8-12 is 2 bytes after -> all merge with gap>=2
        runs = _coalesce_members(
            self._members((0, 4), (6, 10)), max_gap=2, max_bytes=1 << 20
        )
        assert len(runs) == 1
        assert [(m.start, m.end) for m in runs[0]] == [(0, 4), (6, 10)]

    def test_splits_on_large_gap(self):
        runs = _coalesce_members(
            self._members((0, 4), (100, 104)), max_gap=8, max_bytes=1 << 20
        )
        assert [[(m.start, m.end) for m in run] for run in runs] == [
            [(0, 4)],
            [(100, 104)],
        ]

    def test_splits_when_span_exceeds_max_bytes(self):
        runs = _coalesce_members(
            self._members((0, 4), (50, 54)), max_gap=1000, max_bytes=40
        )
        assert len(runs) == 2

    def test_oversized_single_member_is_its_own_run(self):
        runs = _coalesce_members(self._members((0, 1000)), max_gap=8, max_bytes=100)
        assert len(runs) == 1
        assert runs[0][0].end == 1000

    def test_sorts_by_start(self):
        runs = _coalesce_members(
            self._members((8, 12), (0, 4), (4, 8)), max_gap=0, max_bytes=1 << 20
        )
        # all adjacent -> one run, ordered by start
        assert [(m.start, m.end) for m in runs[0]] == [(0, 4), (4, 8), (8, 12)]
