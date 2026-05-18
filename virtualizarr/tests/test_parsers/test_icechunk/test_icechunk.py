"""Tests for IcechunkParser — converts an icechunk repository into a VZ ManifestStore.

Grouped by entry point:

- :class:`TestParseSession` covers ``parse_session(session, registry, ...)``,
  the escape-hatch path for callers that already have an open icechunk
  ``Session``. Also holds the read-then-writeback regression test.
- :class:`TestCall` covers ``__call__(url, registry)``, the protocol-conformant
  path that opens its own ``Repository`` and ``Session`` from a URL.

The obstore -> icechunk.Storage translation used by ``__call__`` lives in a
sibling test file, ``test_icechunk_obstore_utils.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from obspec_utils.registry import ObjectStoreRegistry

# IcechunkParser needs IcechunkStore.array_chunk_iterator, added in 2.0.5.
# Older icechunk is still supported by the writer, so the [icechunk] extra
# pins >=2.0.3 and the parser feature-detects at construction time — skip
# this whole module rather than fail in envs (e.g. pixi minimum-versions)
# that pin icechunk to the writer's floor.
icechunk = pytest.importorskip("icechunk", minversion="2.0.5")
zarr = pytest.importorskip("zarr")
obstore_store = pytest.importorskip("obstore.store")

from virtualizarr.manifests import ManifestArray, ManifestStore
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.parsers import IcechunkParser
from virtualizarr.parsers.typing import Parser


def test_parser_satisfies_parser_protocol() -> None:
    """IcechunkParser must be a runtime-checkable Parser so it works with open_virtual_dataset."""
    assert isinstance(IcechunkParser(), Parser)


# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


def _make_repo(
    storage: icechunk.Storage, *, force_native: bool = False
) -> icechunk.Repository:
    """Create an icechunk repo, optionally with the inline threshold disabled.

    ``force_native=True`` sets ``inline_chunk_threshold_bytes=0`` so every
    chunk write lands as a native (managed) chunk instead of being inlined —
    used by tests that need to inspect native-chunk path rendering.
    """
    config = icechunk.RepositoryConfig.default()
    if force_native:
        config.inline_chunk_threshold_bytes = 0
    return icechunk.Repository.create(storage=storage, config=config)


@pytest.fixture
def mixed_icechunk_repo() -> icechunk.Repository:
    """In-memory icechunk repo with inline + virtual + missing chunks on /a."""
    repo = _make_repo(icechunk.in_memory_storage())
    session = repo.writable_session("main")
    store = session.store

    group = zarr.group(store=store, overwrite=True)
    arr = group.create_array(
        "a",
        shape=(4,),
        chunks=(1,),
        dtype="i4",
        compressors=None,
        dimension_names=("x",),
    )
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
    repo = _make_repo(icechunk.local_filesystem_storage(str(repo_path)))
    session = repo.writable_session("main")
    group = zarr.group(store=session.store, overwrite=True)
    arr = group.create_array("a", shape=(2,), chunks=(1,), dtype="i4", compressors=None)
    arr[0] = 7
    session.commit("init")
    return repo_path, repo


# ----------------------------------------------------------------------
# parse_session: escape-hatch path
# ----------------------------------------------------------------------


class TestParseSession:
    """``parse_session(session, registry, *, native_chunks_prefix)`` entry point."""

    def test_array_metadata_and_manifest(
        self, mixed_icechunk_repo: icechunk.Repository
    ) -> None:
        session = mixed_icechunk_repo.readonly_session(branch="main")
        ms = IcechunkParser().parse_session(
            session,
            registry=ObjectStoreRegistry({}),
            native_chunks_prefix="s3://bucket/repo/chunks",
        )

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
    def test_native_chunks_prefix_applied(self, prefix: str) -> None:
        """Native chunks come back with the user-supplied URL prefix."""
        repo = _make_repo(icechunk.in_memory_storage(), force_native=True)
        session = repo.writable_session("main")
        group = zarr.group(store=session.store, overwrite=True)
        arr = group.create_array(
            "v", shape=(2,), chunks=(1,), dtype="i4", compressors=None
        )
        arr[0] = 5
        arr[1] = 6
        session.commit("c")

        session = repo.readonly_session(branch="main")
        ms = IcechunkParser().parse_session(
            session, registry=ObjectStoreRegistry({}), native_chunks_prefix=prefix
        )
        cm = ms._group.arrays["v"]._manifest

        expected_prefix = "s3://mybucket/myrepo/chunks/"
        for i in (0, 1):
            assert cm._paths[i].startswith(expected_prefix)
            suffix = cm._paths[i].removeprefix(expected_prefix)
            assert suffix and "/" not in suffix
            assert "//" not in cm._paths[i].removeprefix("s3://")
            assert int(cm._lengths[i]) == 4

    def test_skip_variables(self, mixed_icechunk_repo: icechunk.Repository) -> None:
        session = mixed_icechunk_repo.readonly_session(branch="main")
        ms = IcechunkParser(skip_variables=["a"]).parse_session(
            session,
            registry=ObjectStoreRegistry({}),
            native_chunks_prefix="s3://bucket/repo/chunks",
        )
        assert "a" not in ms._group.arrays

    def test_requires_native_chunks_prefix(
        self, mixed_icechunk_repo: icechunk.Repository
    ) -> None:
        """Verify ``native_chunks_prefix`` is a required kwarg on parse_session."""
        session = mixed_icechunk_repo.readonly_session(branch="main")
        with pytest.raises(TypeError):
            IcechunkParser().parse_session(  # type: ignore[call-arg]
                session, registry=ObjectStoreRegistry({})
            )

    def test_round_trip_to_icechunk_writes_valid_chunk_keys(
        self, mixed_icechunk_repo: icechunk.Repository
    ) -> None:
        """Read with IcechunkParser, write back via ``vds.vz.to_icechunk``.

        Regression test for the writer's inline-chunk key encoding: prior to
        the fix the writer used the manifest metadata's separator (``.``) to
        build chunk keys, which icechunk rejected with
        ``invalid zarr key format``. ZarrParser doesn't surface this because
        zarr stores don't produce inline chunks; IcechunkParser is the first
        parser that does.
        """
        session = mixed_icechunk_repo.readonly_session(branch="main")
        ms = IcechunkParser().parse_session(
            session,
            registry=ObjectStoreRegistry({}),
            native_chunks_prefix="s3://bucket/repo/chunks",
        )
        vds = ms.to_virtual_dataset()

        dst_repo = icechunk.Repository.create(storage=icechunk.in_memory_storage())
        dst_session = dst_repo.writable_session("main")
        # Must not raise: IcechunkError("invalid zarr key format `...c.0`").
        vds.vz.to_icechunk(dst_session.store, validate_containers=False)
        dst_session.commit("round-trip")


# ----------------------------------------------------------------------
# __call__: protocol-conformant URL path
# ----------------------------------------------------------------------


class TestCall:
    """``__call__(url, registry)`` entry point — opens repo + session from URL."""

    def test_opens_repo_and_parses(
        self, local_icechunk_repo: tuple[Path, icechunk.Repository]
    ) -> None:
        repo_path, _ = local_icechunk_repo
        parser_root = repo_path.parent
        registry = ObjectStoreRegistry(
            {
                f"file://{parser_root}/": obstore_store.LocalStore(
                    prefix=str(parser_root)
                )
            }
        )

        ms = IcechunkParser()(
            url=f"file://{parser_root}/{repo_path.name}", registry=registry
        )

        assert isinstance(ms, ManifestStore)
        cm = ms._group.arrays["a"]._manifest
        assert cm.shape_chunk_grid == (2,)
        assert cm._paths[0] == INLINED_CHUNK_PATH
        assert (0,) in cm._inlined

    def test_defaults_native_chunks_prefix_to_url_chunks(
        self, local_icechunk_repo: tuple[Path, icechunk.Repository]
    ) -> None:
        """Native chunks rendered with ``f"{url}/chunks/{id}"`` when no prefix is given."""
        repo_path, _ = local_icechunk_repo
        # Reopen with force-native config so the array we add lands as a native
        # chunk (not inline) and we can inspect its rendered URL.
        config = icechunk.RepositoryConfig.default()
        config.inline_chunk_threshold_bytes = 0
        repo = icechunk.Repository.open(
            storage=icechunk.local_filesystem_storage(str(repo_path)), config=config
        )
        session = repo.writable_session("main")
        group = zarr.group(store=session.store)
        arr = group.create_array(
            "native", shape=(1,), chunks=(1,), dtype="i4", compressors=None
        )
        arr[0] = 1
        session.commit("add native chunk")

        parser_root = repo_path.parent
        registry = ObjectStoreRegistry(
            {
                f"file://{parser_root}/": obstore_store.LocalStore(
                    prefix=str(parser_root)
                )
            }
        )
        url = f"file://{parser_root}/{repo_path.name}"

        ms = IcechunkParser()(url=url, registry=registry)
        cm = ms._group.arrays["native"]._manifest
        assert cm._paths[0].startswith(f"{url}/chunks/")
