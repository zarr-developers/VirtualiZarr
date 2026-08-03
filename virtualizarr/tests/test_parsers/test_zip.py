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
from virtualizarr.parsers.zip import ZIP_METHOD_STORED, parse_zip_index

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
