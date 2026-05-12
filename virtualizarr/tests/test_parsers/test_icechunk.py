"""Tests for IcechunkParser — converts an icechunk Session into a VZ ManifestStore."""

from __future__ import annotations

import pytest
from obspec_utils.registry import ObjectStoreRegistry

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")

from virtualizarr.manifests import ManifestArray, ManifestStore
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.parsers import IcechunkParser


@pytest.fixture
def mixed_icechunk_repo() -> icechunk.Repository:
    """In-memory icechunk repo with inline + virtual + missing chunks on /a."""
    config = icechunk.RepositoryConfig.default()
    repo = icechunk.Repository.create(
        storage=icechunk.in_memory_storage(),
        config=config,
    )
    session = repo.writable_session("main")
    store = session.store

    group = zarr.group(store=store, overwrite=True)
    arr = group.create_array(
        "a", shape=(4,), chunks=(1,), dtype="i4", compressors=None
    )
    # inline at slot 0
    arr[0] = 7
    # virtual at slot 2
    store.set_virtual_ref(
        "a/c/2",
        "s3://bucket/data.nc",
        offset=100,
        length=4,
        validate_container=False,
    )
    # slots 1 and 3 left missing
    session.commit("init")
    return repo


def test_parser_returns_manifest_store(mixed_icechunk_repo: icechunk.Repository) -> None:
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser(native_chunks_prefix="s3://bucket/repo/chunks")
    ms = parser(session.store, registry=ObjectStoreRegistry({}))
    assert isinstance(ms, ManifestStore)


def test_parser_array_metadata_and_manifest(
    mixed_icechunk_repo: icechunk.Repository,
) -> None:
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser(native_chunks_prefix="s3://bucket/repo/chunks")
    ms = parser(session.store, registry=ObjectStoreRegistry({}))

    ma = ms._group.arrays["a"]
    assert isinstance(ma, ManifestArray)
    assert ma.shape == (4,)

    cm = ma._manifest
    assert cm.shape_chunk_grid == (4,)

    # Inline slot 0
    assert cm._paths[0] == INLINED_CHUNK_PATH
    assert cm._lengths[0] == 4
    assert (0,) in cm._inlined
    assert len(cm._inlined[(0,)]) == 4

    # Virtual slot 2
    assert cm._paths[2] == "s3://bucket/data.nc"
    assert int(cm._offsets[2]) == 100
    assert int(cm._lengths[2]) == 4

    # Missing slots 1 and 3
    for i in (1, 3):
        assert cm._paths[i] == ""
        assert int(cm._lengths[i]) == 0


def test_parser_native_chunks_prefix_applied() -> None:
    """Native chunks must come back with the user-supplied URL prefix."""
    config = icechunk.RepositoryConfig.default()
    config.inline_chunk_threshold_bytes = 0  # force every write to be native
    repo = icechunk.Repository.create(
        storage=icechunk.in_memory_storage(),
        config=config,
    )
    session = repo.writable_session("main")
    group = zarr.group(store=session.store, overwrite=True)
    arr = group.create_array("v", shape=(2,), chunks=(1,), dtype="i4", compressors=None)
    arr[0] = 5
    arr[1] = 6
    session.commit("c")

    session = repo.readonly_session(branch="main")
    parser = IcechunkParser(native_chunks_prefix="s3://mybucket/myrepo/chunks")
    ms = parser(session.store, registry=ObjectStoreRegistry({}))
    cm = ms._group.arrays["v"]._manifest

    for i in (0, 1):
        assert cm._paths[i].startswith("s3://mybucket/myrepo/chunks/")
        # bare chunk_id sits between the prefix and end — non-empty, no further slashes
        suffix = cm._paths[i].removeprefix("s3://mybucket/myrepo/chunks/")
        assert suffix
        assert "/" not in suffix
        assert int(cm._lengths[i]) == 4


def test_parser_native_chunks_prefix_trailing_slash_tolerated() -> None:
    """A trailing slash on the prefix shouldn't produce '//' in chunk URLs."""
    config = icechunk.RepositoryConfig.default()
    config.inline_chunk_threshold_bytes = 0
    repo = icechunk.Repository.create(
        storage=icechunk.in_memory_storage(),
        config=config,
    )
    session = repo.writable_session("main")
    group = zarr.group(store=session.store, overwrite=True)
    arr = group.create_array("v", shape=(1,), chunks=(1,), dtype="i4", compressors=None)
    arr[0] = 1
    session.commit("c")

    session = repo.readonly_session(branch="main")
    parser = IcechunkParser(native_chunks_prefix="s3://mybucket/myrepo/chunks/")
    ms = parser(session.store, registry=ObjectStoreRegistry({}))
    cm = ms._group.arrays["v"]._manifest

    assert "//" not in cm._paths[0].removeprefix("s3://")


def test_parser_skip_variables(mixed_icechunk_repo: icechunk.Repository) -> None:
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser(
        native_chunks_prefix="s3://bucket/repo/chunks",
        skip_variables=["a"],
    )
    ms = parser(session.store, registry=ObjectStoreRegistry({}))
    assert "a" not in ms._group.arrays
