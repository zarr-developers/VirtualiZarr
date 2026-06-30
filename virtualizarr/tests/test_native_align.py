"""
End-to-end tests that native xarray alignment machinery works over ManifestArrays.

The point of the indexer/where/result_type hooks is that users call plain
`xr.reindex`, `xr.align`, and `xr.concat` — no VirtualiZarr-specific API — and the
fill stays virtual (null-path chunks that read back as the array's fill_value).
"""

import numpy as np
import obstore as obs
import pytest
import xarray as xr
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata

CODECS = [{"configuration": {"endian": "little"}, "name": "bytes"}]
F32 = {
    1.0: b"\x00\x00\x80\x3f",
    2.0: b"\x00\x00\x00\x40",
    3.0: b"\x00\x00\x40\x40",
}
I32 = {10: b"\x0a\x00\x00\x00", 20: b"\x14\x00\x00\x00", 30: b"\x1e\x00\x00\x00"}


def _ds_1d(x, path, dtype="float32", fill_value=float("nan"), chunk=1, name="foo"):
    n = len(x)
    n_chunks = -(-n // chunk)
    entries = {
        str(i): {"path": path, "offset": i * chunk * 4, "length": chunk * 4}
        for i in range(n_chunks)
    }
    metadata = create_v3_array_metadata(
        shape=(n,),
        chunk_shape=(chunk,),
        data_type=np.dtype(dtype),
        codecs=CODECS,
        fill_value=fill_value,
        dimension_names=["x"],
    )
    marr = ManifestArray(
        metadata=metadata, chunkmanifest=ChunkManifest(entries=entries)
    )
    return xr.Dataset({name: xr.Variable(["x"], marr)}, coords={"x": list(x)})


def _ds_2d(x, t, path, name="foo"):
    nt, nx = len(t), len(x)
    entries = {}
    k = 0
    for it in range(nt):
        for ix in range(nx):
            entries[f"{it}.{ix}"] = {"path": path, "offset": k * 4, "length": 4}
            k += 1
    metadata = create_v3_array_metadata(
        shape=(nt, nx),
        chunk_shape=(1, 1),
        data_type=np.dtype("float32"),
        codecs=CODECS,
        fill_value=float("nan"),
        dimension_names=["time", "x"],
    )
    marr = ManifestArray(
        metadata=metadata, chunkmanifest=ChunkManifest(entries=entries)
    )
    return xr.Dataset(
        {name: xr.Variable(["time", "x"], marr)}, coords={"x": list(x), "time": list(t)}
    )


class TestNativeReindex:
    def test_reindex_stays_lazy(self):
        ds = _ds_1d([0, 1, 2], "/a.nc")
        result = ds.reindex(x=[0, 1, 2, 3, 4])  # plain xarray, no vz API

        assert isinstance(result["foo"].data, ManifestArray)
        assert list(result.x.values) == [0, 1, 2, 3, 4]
        assert result["foo"].data.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 0, "length": 4},
            "1": {"path": "file:///a.nc", "offset": 4, "length": 4},
            "2": {"path": "file:///a.nc", "offset": 8, "length": 4},
        }

    def test_reindex_float_reads_back_nan(self):
        store = obs.store.MemoryStore()
        obs.put(store, "a.bin", F32[1.0] + F32[2.0] + F32[3.0])
        ds = _ds_1d([0, 1, 2], "memory:///a.bin")

        reindexed = ds.reindex(x=[0, 1, 2, 3, 4])

        ms = ManifestStore(
            group=ManifestGroup(arrays={"foo": reindexed["foo"].data}),
            registry=ObjectStoreRegistry({"memory://": store}),
        )
        vals = xr.open_zarr(ms, consolidated=False, zarr_format=3)["foo"].values
        np.testing.assert_array_equal(vals[:3], [1.0, 2.0, 3.0])
        assert np.isnan(vals[3:]).all()

    # xarray internally builds a NaN fill scalar and casts it toward the int
    # dtype (a value we never use, since our np.where returns the array as-is),
    # which emits a harmless "invalid value encountered in cast" RuntimeWarning.
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in cast:RuntimeWarning"
    )
    def test_reindex_int_keeps_dtype_and_sentinel(self):
        # xarray would normally promote int->float+NaN; the manifest keeps int32
        # and its declared sentinel fill instead.
        store = obs.store.MemoryStore()
        obs.put(store, "a.bin", I32[10] + I32[20] + I32[30])
        ds = _ds_1d([0, 1, 2], "memory:///a.bin", dtype="int32", fill_value=-9999)

        reindexed = ds.reindex(x=[0, 1, 2, 3, 4])
        assert reindexed["foo"].dtype == np.dtype("int32")
        assert isinstance(reindexed["foo"].data, ManifestArray)

        ms = ManifestStore(
            group=ManifestGroup(arrays={"foo": reindexed["foo"].data}),
            registry=ObjectStoreRegistry({"memory://": store}),
        )
        vals = xr.open_zarr(ms, consolidated=False, zarr_format=3)["foo"].values
        np.testing.assert_array_equal(vals, [10, 20, 30, -9999, -9999])
        assert vals.dtype == np.dtype("int32")

    def test_reindex_non_chunk_aligned_raises(self):
        ds = _ds_1d([0, 1, 2], "/a.nc", chunk=2)
        # the deep error is wrapped with call-site context: the operation, the
        # axis, and what to do — and the low-level reason is chained.
        with pytest.raises(
            NotImplementedError, match="align/reindex this virtual array along axis 0"
        ) as excinfo:
            ds.reindex(x=[0, 1, 99, 2])
        assert "chunk boundaries" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, NotImplementedError)


class TestNativeReindexMultiAxis:
    def test_reindex_two_axes_simultaneously(self):
        # A 2D outer-join reindexes BOTH axes in one call. xarray then sends a
        # single broadcast (vectorized) indexer -- one N-D array per axis, shaped
        # (Nt, 1) and (1, Nx) -- rather than two 1D indexers. The chunk-grid remap
        # must collapse each broadcast component back to its 1D per-axis indexer,
        # or it misreads the array's rank (the real ITS_LIVE x+y mosaic case).
        store = obs.store.MemoryStore()
        # 2x2 grid, row-major (time, x): (t0,x0)=1 (t0,x1)=2 (t1,x0)=3 (t1,x1)=1
        obs.put(store, "a.bin", F32[1.0] + F32[2.0] + F32[3.0] + F32[1.0])
        ds = _ds_2d([0, 1], [0, 1], "memory:///a.bin")  # x=[0,1], time=[0,1]

        result = ds.reindex(time=[0, 1, 2], x=[0, 1, 2])

        assert isinstance(result["foo"].data, ManifestArray)
        assert list(result.time.values) == [0, 1, 2]
        assert list(result.x.values) == [0, 1, 2]

        ms = ManifestStore(
            group=ManifestGroup(arrays={"foo": result["foo"].data}),
            registry=ObjectStoreRegistry({"memory://": store}),
        )
        vals = xr.open_zarr(ms, consolidated=False, zarr_format=3)["foo"].values
        np.testing.assert_array_equal(vals[:2, :2], [[1.0, 2.0], [3.0, 1.0]])
        assert np.isnan(vals[2, :]).all()  # appended time row reads back as fill
        assert np.isnan(vals[:, 2]).all()  # appended x column reads back as fill


class TestNativeAlign:
    def test_align_outer_union(self):
        a = _ds_1d([0, 1, 2], "/a.nc")
        b = _ds_1d([3, 4, 5], "/b.nc")

        ra, rb = xr.align(a, b, join="outer")

        assert isinstance(ra["foo"].data, ManifestArray)
        assert isinstance(rb["foo"].data, ManifestArray)
        assert list(ra.x.values) == [0, 1, 2, 3, 4, 5]

    def test_align_exclude_then_concat_spatial(self):
        # the ITS_LIVE pattern: align on the spatial dim (exclude time), concat on time
        store = obs.store.MemoryStore()
        obs.put(store, "a.bin", F32[1.0] + F32[2.0])  # time=0, x=0,1
        obs.put(store, "b.bin", F32[3.0] + F32[1.0])  # time=1, x=2,3
        a = _ds_2d([0, 1], [0], "memory:///a.bin")
        b = _ds_2d([2, 3], [1], "memory:///b.bin")

        ra, rb = xr.align(a, b, join="outer", exclude=["time"])
        # time is left alone; x is unioned
        assert list(ra.time.values) == [0]
        assert list(rb.time.values) == [1]
        assert list(ra.x.values) == [0, 1, 2, 3]

        cube = xr.concat([ra, rb], dim="time", join="exact")
        assert cube.sizes == {"time": 2, "x": 4}
        assert isinstance(cube["foo"].data, ManifestArray)

        ms = ManifestStore(
            group=ManifestGroup(arrays={"foo": cube["foo"].data}),
            registry=ObjectStoreRegistry({"memory://": store}),
        )
        vals = xr.open_zarr(ms, consolidated=False, zarr_format=3)["foo"].values
        np.testing.assert_array_equal(vals[0, :2], [1.0, 2.0])
        assert np.isnan(vals[0, 2:]).all()
        assert np.isnan(vals[1, :2]).all()
        np.testing.assert_array_equal(vals[1, 2:], [3.0, 1.0])


class TestGeneralWhereStillRejected:
    def test_boolean_masking_raises_clearly(self):
        # we must NOT silently no-op general where(); it should raise, since it
        # would require materializing values.
        ds = _ds_1d([0, 1, 2], "/a.nc")
        with pytest.raises(NotImplementedError, match="materializing|where"):
            ds["foo"].where(ds["x"] > 0).compute()
