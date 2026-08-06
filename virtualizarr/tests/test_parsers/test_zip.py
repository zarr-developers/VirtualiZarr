import asyncio
import zipfile

import numpy as np
import pytest
import xarray as xr
import zarr
from obstore.store import LocalStore
from packaging import version
from zarr.storage import ZipStore

from virtualizarr import open_virtual_dataset, open_virtual_datatree
from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import ZippedZarrParser
from virtualizarr.parsers.zarr.zip import ZIP_METHOD_STORED, parse_zip_index

requires_v2_migration = pytest.mark.skipif(
    version.parse(zarr.__version__) < version.parse("3.1.3"),
    reason="V2→V3 metadata migration requires zarr>=3.1.3",
)

zarr_versions = pytest.mark.parametrize(
    "zarr_format",
    [
        pytest.param(2, id="Zarr V2", marks=requires_v2_migration),
        pytest.param(3, id="Zarr V3"),
    ],
)


def _test_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"air": (("time", "lat", "lon"), rng.random((4, 3, 2)))},
        coords={
            "time": np.arange(4, dtype="int64"),
            "lat": np.array([10.0, 20.0, 30.0]),
            "lon": np.array([1.0, 2.0]),
        },
        attrs={"description": "zipped zarr test data"},
    )


def _write_zipped_zarr(zip_path, ds, zarr_format=3, **kwargs) -> None:
    store = ZipStore(zip_path, mode="w")
    try:
        ds.to_zarr(
            store,
            zarr_format=zarr_format,
            consolidated=False,
            encoding={"air": {"chunks": (2, 3, 2)}},
            **kwargs,
        )
    finally:
        store.close()


@zarr_versions
def test_roundtrip(tmp_path, local_registry, zarr_format):
    ds = _test_dataset()
    zip_path = tmp_path / "air.zarr.zip"
    _write_zipped_zarr(zip_path, ds, zarr_format)

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
        loadable_variables=list(ds.variables),
    ) as vds:
        xr.testing.assert_identical(vds, ds)


@zarr_versions
def test_manifest_points_into_archive(tmp_path, local_registry, zarr_format):
    ds = _test_dataset()
    zip_path = tmp_path / "air.zarr.zip"
    _write_zipped_zarr(zip_path, ds, zarr_format)

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
        loadable_variables=[],
    ) as vds:
        marr = vds["air"].data
        assert isinstance(marr, ManifestArray)
        manifest = marr.manifest.dict()
        assert set(manifest.keys()) == {"0.0.0", "1.0.0"}

        # every chunk must be a byte range within the archive itself
        raw = zip_path.read_bytes()
        chunk_member = "air/c/0/0/0" if zarr_format == 3 else "air/0.0.0"
        with zipfile.ZipFile(zip_path) as zf:
            expected = zf.read(chunk_member)
        entry = manifest["0.0.0"]
        assert entry["path"].endswith("air.zarr.zip")
        assert entry["offset"] > 0
        assert raw[entry["offset"] : entry["offset"] + entry["length"]] == expected


def test_parse_zip_index_matches_zipfile(tmp_path):
    ds = _test_dataset()
    zip_path = tmp_path / "air.zarr.zip"
    _write_zipped_zarr(zip_path, ds)

    store = LocalStore(prefix=str(tmp_path))
    index = asyncio.run(parse_zip_index(store, "air.zarr.zip"))

    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        names = [zi.filename for zi in zf.infolist() if not zi.is_dir()]
        assert set(index.keys()) == set(names)
        for name in names:
            entry = index[name]
            assert entry.method == ZIP_METHOD_STORED
            assert raw[
                entry.data_offset : entry.data_offset + entry.compressed_length
            ] == zf.read(name)


def test_deflate_raises(tmp_path, local_registry):
    ds = _test_dataset()
    zip_path = tmp_path / "air.zarr.zip"
    store = ZipStore(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED)
    try:
        ds.to_zarr(store, zarr_format=3, consolidated=False)
    finally:
        store.close()

    with pytest.raises(NotImplementedError, match="STORED"):
        open_virtual_dataset(
            url=f"file://{zip_path}",
            registry=local_registry,
            parser=ZippedZarrParser(),
        )


def test_not_a_zip_raises(tmp_path, local_registry):
    path = tmp_path / "not_a.zarr.zip"
    path.write_bytes(b"this is not a zip archive" * 10)

    with pytest.raises(ValueError, match="not a zip archive"):
        open_virtual_dataset(
            url=f"file://{path}",
            registry=local_registry,
            parser=ZippedZarrParser(),
        )


def test_datatree(tmp_path, local_registry):
    ds = _test_dataset()
    child = xr.Dataset({"pressure": (("y",), np.arange(3, dtype="float64"))})
    dt = xr.DataTree.from_dict({"/": ds, "/child": child})
    zip_path = tmp_path / "tree.zarr.zip"
    store = ZipStore(zip_path, mode="w")
    try:
        dt.to_zarr(store, zarr_format=3, consolidated=False)
    finally:
        store.close()

    with open_virtual_datatree(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
    ) as vdt:
        assert "child" in vdt.children
        assert "air" in vdt.ds
        assert "pressure" in vdt["child"].ds


def test_group_kwarg(tmp_path, local_registry):
    ds = _test_dataset()
    child = xr.Dataset({"pressure": (("y",), np.arange(3, dtype="float64"))})
    dt = xr.DataTree.from_dict({"/": ds, "/child": child})
    zip_path = tmp_path / "tree.zarr.zip"
    store = ZipStore(zip_path, mode="w")
    try:
        dt.to_zarr(store, zarr_format=3, consolidated=False)
    finally:
        store.close()

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(group="child"),
        loadable_variables=["pressure"],
    ) as vds:
        xr.testing.assert_identical(vds["pressure"], child["pressure"])


def test_skip_variables(tmp_path, local_registry):
    ds = _test_dataset()
    zip_path = tmp_path / "air.zarr.zip"
    _write_zipped_zarr(zip_path, ds)

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(skip_variables=["air"]),
    ) as vds:
        assert len(vds.data_vars) == 0


def test_scalar_array(tmp_path, local_registry):
    ds = xr.Dataset({"scalar": ((), np.float64(3.14))})
    zip_path = tmp_path / "scalar.zarr.zip"
    store = ZipStore(zip_path, mode="w")
    try:
        ds.to_zarr(store, zarr_format=3, consolidated=False)
    finally:
        store.close()

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
        loadable_variables=["scalar"],
    ) as vds:
        xr.testing.assert_identical(vds, ds)


def test_v3_dot_chunk_key_separator(tmp_path, local_registry):
    """Zarr V3 arrays may use a "." chunk key separator, giving members like
    "air/c.0.0" rather than "air/c/0/0". Regression test for #1069."""
    ds = _test_dataset()
    zip_path = tmp_path / "dot_sep.zarr.zip"
    store = ZipStore(zip_path, mode="w")
    try:
        ds.to_zarr(
            store,
            zarr_format=3,
            consolidated=False,
            encoding={
                "air": {
                    "chunks": (2, 3, 2),
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "."},
                    },
                }
            },
        )
    finally:
        store.close()

    with zipfile.ZipFile(zip_path) as zf:
        assert "air/c.0.0.0" in zf.namelist()

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
        loadable_variables=[],
    ) as vds:
        assert set(vds["air"].data.manifest.dict()) == {"0.0.0", "1.0.0"}

    with open_virtual_dataset(
        url=f"file://{zip_path}",
        registry=local_registry,
        parser=ZippedZarrParser(),
        loadable_variables=list(ds.variables),
    ) as vds:
        xr.testing.assert_identical(vds, ds)


class _CountingStore:
    """Wraps an obstore, counting the range requests the parser makes."""

    def __init__(self, store):
        self._store = store
        self.n_ranges = 0

    async def head_async(self, path):
        return await self._store.head_async(path)

    async def get_range_async(self, path, *, start, end):
        self.n_ranges += 1
        return await self._store.get_range_async(path, start=start, end=end)

    async def get_ranges_async(self, path, *, starts, ends):
        self.n_ranges += len(starts)
        return await self._store.get_ranges_async(path, starts=starts, ends=ends)


@zarr_versions
def test_index_does_not_read_local_headers(tmp_path, zarr_format):
    """
    A zipped Zarr's member offsets are derivable from the central directory alone, so
    building the index must not cost a read per member.

    Reading every 30-byte local header is one request per member - hundreds for a real
    archive - which dominates the cost of virtualizing it over a high-latency store.
    """
    ds = xr.Dataset({f"var_{i}": (("x",), np.arange(8, dtype="f4")) for i in range(30)})
    zip_path = tmp_path / f"many_v{zarr_format}.zarr.zip"
    store = ZipStore(zip_path, mode="w")
    try:
        ds.to_zarr(store, zarr_format=zarr_format, consolidated=False)
    finally:
        store.close()

    counting = _CountingStore(LocalStore(prefix=str(tmp_path)))
    index = asyncio.run(parse_zip_index(counting, zip_path.name))

    assert len(index) > 30  # metadata + chunks
    assert counting.n_ranges <= 3, counting.n_ranges

    # and the offsets must still be right
    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        for name, entry in index.items():
            assert raw[
                entry.data_offset : entry.data_offset + entry.compressed_length
            ] == zf.read(name), name


def test_falls_back_to_local_headers_when_offsets_disagree(tmp_path):
    """
    Archives whose local extra fields differ from the central directory's must still be
    read correctly, by falling back to the local headers.

    `force_zip64` is the documented divergence: zipfile writes a 20-byte ZIP64 extra field
    into the local header but not into the central directory.
    """
    zip_path = tmp_path / "forced.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(3):
            with zf.open(
                zipfile.ZipInfo(f"member_{i}.bin"), "w", force_zip64=True
            ) as f:
                f.write(bytes([i]) * 500)

    counting = _CountingStore(LocalStore(prefix=str(tmp_path)))
    index = asyncio.run(parse_zip_index(counting, zip_path.name))

    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        for name, entry in index.items():
            assert raw[
                entry.data_offset : entry.data_offset + entry.compressed_length
            ] == zf.read(name), name

    # the fallback means it paid for the local headers, rather than being wrong
    assert counting.n_ranges > 3, counting.n_ranges
