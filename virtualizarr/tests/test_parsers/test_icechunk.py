"""Tests for IcechunkParser — converts an icechunk repository into a VZ ManifestStore."""

from __future__ import annotations

from pathlib import Path

import pytest
from obspec_utils.registry import ObjectStoreRegistry

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")
obstore_store = pytest.importorskip("obstore.store")

from virtualizarr.manifests import ManifestArray, ManifestStore
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.parsers import IcechunkParser
from virtualizarr.parsers.typing import Parser


def test_parser_satisfies_parser_protocol() -> None:
    """IcechunkParser must be a runtime-checkable Parser so it works with open_virtual_dataset."""
    assert isinstance(IcechunkParser(native_chunks_prefix="s3://x/y"), Parser)


# ----------------------------------------------------------------------
# Fixtures: in-memory repo (used by parse_session path) and on-disk repo
# (used by the protocol-conformant __call__ path).
# ----------------------------------------------------------------------


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
    arr = group.create_array("a", shape=(4,), chunks=(1,), dtype="i4", compressors=None)
    arr[0] = 7  # inline
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


@pytest.fixture
def local_icechunk_repo(tmp_path: Path) -> tuple[Path, icechunk.Repository]:
    """Disk-backed icechunk repo with one inline chunk on /a. Returns (path, repo)."""
    repo_path = tmp_path / "repo"
    repo = icechunk.Repository.create(
        storage=icechunk.local_filesystem_storage(str(repo_path)),
    )
    session = repo.writable_session("main")
    group = zarr.group(store=session.store, overwrite=True)
    arr = group.create_array("a", shape=(2,), chunks=(1,), dtype="i4", compressors=None)
    arr[0] = 7
    session.commit("init")
    return repo_path, repo


# ----------------------------------------------------------------------
# parse_session: escape-hatch path
# ----------------------------------------------------------------------


def test_parse_session_array_metadata_and_manifest(
    mixed_icechunk_repo: icechunk.Repository,
) -> None:
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser(native_chunks_prefix="s3://bucket/repo/chunks")
    ms = parser.parse_session(session, registry=ObjectStoreRegistry({}))

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


@pytest.mark.parametrize(
    "prefix",
    [
        "s3://mybucket/myrepo/chunks",
        "s3://mybucket/myrepo/chunks/",  # trailing slash must be tolerated
    ],
)
def test_parse_session_native_chunks_prefix_applied(prefix: str) -> None:
    """Native chunks come back with the user-supplied URL prefix."""
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
    parser = IcechunkParser(native_chunks_prefix=prefix)
    ms = parser.parse_session(session, registry=ObjectStoreRegistry({}))
    cm = ms._group.arrays["v"]._manifest

    expected_prefix = "s3://mybucket/myrepo/chunks/"
    for i in (0, 1):
        assert cm._paths[i].startswith(expected_prefix)
        suffix = cm._paths[i].removeprefix(expected_prefix)
        assert suffix and "/" not in suffix
        assert "//" not in cm._paths[i].removeprefix("s3://")
        assert int(cm._lengths[i]) == 4


def test_parse_session_skip_variables(mixed_icechunk_repo: icechunk.Repository) -> None:
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser(
        native_chunks_prefix="s3://bucket/repo/chunks",
        skip_variables=["a"],
    )
    ms = parser.parse_session(session, registry=ObjectStoreRegistry({}))
    assert "a" not in ms._group.arrays


# ----------------------------------------------------------------------
# __call__: protocol-conformant URL path
# ----------------------------------------------------------------------


def test_call_via_url_opens_repo_and_parses(
    local_icechunk_repo: tuple[Path, icechunk.Repository],
) -> None:
    """The Protocol-conformant path: __call__(url, registry) opens icechunk itself."""
    repo_path, _ = local_icechunk_repo
    parser_root = repo_path.parent
    repo_name = repo_path.name

    registry = ObjectStoreRegistry(
        {f"file://{parser_root}/": obstore_store.LocalStore(prefix=str(parser_root))}
    )

    # native_chunks_prefix not given — should default to f"{url}/chunks".
    parser = IcechunkParser()
    ms = parser(url=f"file://{parser_root}/{repo_name}", registry=registry)

    assert isinstance(ms, ManifestStore)
    cm = ms._group.arrays["a"]._manifest
    assert cm.shape_chunk_grid == (2,)
    assert cm._paths[0] == INLINED_CHUNK_PATH
    assert (0,) in cm._inlined


def test_call_defaults_native_chunks_prefix_to_url_chunks(
    local_icechunk_repo: tuple[Path, icechunk.Repository],
) -> None:
    """Native chunks rendered with `{url}/chunks/{id}` when the prefix is implicit."""
    repo_path, repo = local_icechunk_repo
    # Write an extra array with native chunks so we can inspect the prefix.
    config = icechunk.RepositoryConfig.default()
    config.inline_chunk_threshold_bytes = 0
    # Reopen the same on-disk repo with a config that forces native chunks.
    repo = icechunk.Repository.open(
        storage=icechunk.local_filesystem_storage(str(repo_path)),
        config=config,
    )
    session = repo.writable_session("main")
    group = zarr.group(store=session.store)
    arr = group.create_array(
        "native", shape=(1,), chunks=(1,), dtype="i4", compressors=None
    )
    arr[0] = 1
    session.commit("add native chunk")

    parser_root = repo_path.parent
    repo_name = repo_path.name
    registry = ObjectStoreRegistry(
        {f"file://{parser_root}/": obstore_store.LocalStore(prefix=str(parser_root))}
    )

    parser = IcechunkParser()
    url = f"file://{parser_root}/{repo_name}"
    ms = parser(url=url, registry=registry)
    cm = ms._group.arrays["native"]._manifest
    assert cm._paths[0].startswith(f"{url}/chunks/")


def test_parse_session_requires_explicit_native_chunks_prefix(
    mixed_icechunk_repo: icechunk.Repository,
) -> None:
    """parse_session can't infer the prefix — the user must supply it."""
    session = mixed_icechunk_repo.readonly_session(branch="main")
    parser = IcechunkParser()  # no prefix
    with pytest.raises(ValueError, match="native_chunks_prefix"):
        parser.parse_session(session, registry=ObjectStoreRegistry({}))
