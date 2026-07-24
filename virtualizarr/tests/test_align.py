"""
End-to-end tests that xarray alignment machinery works over ManifestArrays.

Users call plain ``xr.reindex`` / ``xr.align`` / ``xr.concat`` -- no
VirtualiZarr-specific API -- and the fill stays virtual: null-path chunks that
read back as the array's ``fill_value``.

Structure/laziness cases build ManifestArrays directly via the ``array_v3_metadata``
fixture (as ``TestConcat`` in test_xarray.py does). Value-readback cases open real
tiled NetCDF files via the ``netcdf4_files_factory_2d`` fixture + a parser (as
``TestCombine`` does) and read the reindexed result back through a ManifestStore.
"""

import numpy as np
import pytest
import xarray as xr

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import copy_and_replace_metadata
from virtualizarr.parsers import HDFParser
from virtualizarr.tests import requires_hdf5plugin, requires_imagecodecs

LOADABLE_COORDS = ["time", "lat", "lon"]


def _read_back(marr: ManifestArray, dims: list[str], registry) -> np.ndarray:
    """Materialise a virtual ManifestArray's values by wrapping it in a
    ManifestStore and opening it.

    Reindex preserves whatever ``dimension_names`` the source metadata carried;
    parser-built arrays carry ``None``, which ``xr.open_zarr`` requires, so set
    them here for the read.
    """
    metadata = copy_and_replace_metadata(marr.metadata, new_dimension_names=dims)
    ms = ManifestStore(
        group=ManifestGroup(
            arrays={"v": ManifestArray(metadata=metadata, chunkmanifest=marr.manifest)}
        ),
        registry=registry,
    )
    return xr.open_zarr(ms, consolidated=False, zarr_format=3)["v"].values


class TestReindex:
    def test_reindex_stays_lazy(self, array_v3_metadata):
        metadata = array_v3_metadata(shape=(3,), chunks=(1,), dimension_names=["x"])
        marr = ManifestArray(
            metadata=metadata,
            chunkmanifest=ChunkManifest(
                entries={
                    "0": {"path": "/a.nc", "offset": 0, "length": 4},
                    "1": {"path": "/a.nc", "offset": 4, "length": 4},
                    "2": {"path": "/a.nc", "offset": 8, "length": 4},
                }
            ),
        )
        ds = xr.Dataset({"foo": ("x", marr)}, coords={"x": [0, 1, 2]})

        result = ds.reindex(x=[0, 1, 2, 3, 4])  # plain xarray, no vz API

        assert isinstance(result["foo"].data, ManifestArray)
        assert list(result.x.values) == [0, 1, 2, 3, 4]
        # appended positions become null chunks (absent from the manifest)
        assert result["foo"].data.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 0, "length": 4},
            "1": {"path": "file:///a.nc", "offset": 4, "length": 4},
            "2": {"path": "file:///a.nc", "offset": 8, "length": 4},
        }

    def test_reindex_two_axes_simultaneously(self, array_v3_metadata):
        # A 2D outer-join reindexes BOTH axes in one call, which xarray sends as a
        # single broadcast (vectorized) indexer -- one N-D array per axis, shaped
        # (Nt, 1) and (1, Nx) -- rather than two 1D indexers. The chunk-grid remap
        # must collapse each back to its 1D per-axis indexer, else it misreads the
        # array's rank (the real ITS_LIVE x+y mosaic case).
        metadata = array_v3_metadata(
            shape=(2, 2), chunks=(1, 1), dimension_names=["time", "x"]
        )
        marr = ManifestArray(
            metadata=metadata,
            chunkmanifest=ChunkManifest(
                entries={
                    "0.0": {"path": "/a.nc", "offset": 0, "length": 4},
                    "0.1": {"path": "/a.nc", "offset": 4, "length": 4},
                    "1.0": {"path": "/a.nc", "offset": 8, "length": 4},
                    "1.1": {"path": "/a.nc", "offset": 12, "length": 4},
                }
            ),
        )
        ds = xr.Dataset(
            {"foo": (["time", "x"], marr)}, coords={"time": [0, 1], "x": [0, 1]}
        )

        result = ds.reindex(time=[0, 1, 2], x=[0, 1, 2])

        assert isinstance(result["foo"].data, ManifestArray)
        assert result["foo"].shape == (3, 3)
        # the four real chunks are kept; the appended row and column are null
        assert result["foo"].data.manifest.dict() == {
            "0.0": {"path": "file:///a.nc", "offset": 0, "length": 4},
            "0.1": {"path": "file:///a.nc", "offset": 4, "length": 4},
            "1.0": {"path": "file:///a.nc", "offset": 8, "length": 4},
            "1.1": {"path": "file:///a.nc", "offset": 12, "length": 4},
        }

    def test_reindex_non_chunk_aligned_raises(self, array_v3_metadata):
        metadata = array_v3_metadata(shape=(3,), chunks=(2,), dimension_names=["x"])
        marr = ManifestArray(
            metadata=metadata,
            chunkmanifest=ChunkManifest(
                entries={
                    "0": {"path": "/a.nc", "offset": 0, "length": 8},
                    "1": {"path": "/a.nc", "offset": 8, "length": 4},
                }
            ),
        )
        ds = xr.Dataset({"foo": ("x", marr)}, coords={"x": [0, 1, 2]})
        # the deep error is wrapped with call-site context: the operation, the
        # axis, and what to do -- and the low-level reason is chained.
        with pytest.raises(
            NotImplementedError, match="align/reindex this virtual array along axis 0"
        ) as excinfo:
            ds.reindex(x=[0, 1, 99, 2])
        assert "chunk boundaries" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, NotImplementedError)


class TestAlign:
    def test_align_outer_union(self, array_v3_metadata):
        def _ds(path, xs):
            metadata = array_v3_metadata(shape=(3,), chunks=(1,), dimension_names=["x"])
            manifest = ChunkManifest(
                entries={
                    str(i): {"path": path, "offset": i * 4, "length": 4}
                    for i in range(3)
                }
            )
            marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
            return xr.Dataset({"foo": ("x", marr)}, coords={"x": xs})

        a = _ds("/a.nc", [0, 1, 2])
        b = _ds("/b.nc", [3, 4, 5])

        ra, rb = xr.align(a, b, join="outer")

        assert isinstance(ra["foo"].data, ManifestArray)
        assert isinstance(rb["foo"].data, ManifestArray)
        assert list(ra.x.values) == [0, 1, 2, 3, 4, 5]


@requires_hdf5plugin
@requires_imagecodecs
class TestReindexReadback:
    """Reindex fill reads back as the array's fill_value, verified over real files.

    Uses the air_temperature tiles from ``netcdf4_files_factory_2d`` (as
    ``TestCombine`` does): two time slices x two lat bands.

    NOTE: air's ``lat`` is descending, so ``xr.align(join="outer")`` -- which sorts
    the union ascending -- would require reversing within a chunk (not lazy). The
    chunk-aligned, lazy path is an explicit ``reindex`` onto a descending union,
    which is also the real ITS_LIVE mosaic pattern.
    """

    def test_spatial_reindex_then_concat_on_time_keeps_int_sentinel(
        self, netcdf4_files_factory_2d, local_registry
    ):
        # air_temperature is stored int16 with sentinel fill -32767. Two tiles that
        # differ in BOTH time slice and lat band (the ITS_LIVE granule shape):
        # reindex each onto the union lat, then concat on time.
        early_north, _, _, late_south = netcdf4_files_factory_2d()
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=early_north,
                registry=local_registry,
                parser=parser,
                loadable_variables=LOADABLE_COORDS,
            ) as en,
            open_virtual_dataset(
                url=late_south,
                registry=local_registry,
                parser=parser,
                loadable_variables=LOADABLE_COORDS,
            ) as ls,
        ):
            union_lat = np.concatenate([en.lat.values, ls.lat.values])  # descending
            cube = xr.concat(
                [en.reindex(lat=union_lat), ls.reindex(lat=union_lat)],
                dim="time",
                join="exact",
            )
            assert isinstance(cube["air"].data, ManifestArray)
            assert cube["air"].dtype == np.dtype("int16")  # not promoted to float

            nt, nlat = en.sizes["time"], en.sizes["lat"]
            fill = cube["air"].data.metadata.fill_value
            vals = _read_back(cube["air"].data, ["time", "lat", "lon"], local_registry)
            en_vals = _read_back(en["air"].data, ["time", "lat", "lon"], local_registry)
            ls_vals = _read_back(ls["air"].data, ["time", "lat", "lon"], local_registry)

            np.testing.assert_array_equal(vals[:nt, :nlat, :], en_vals)  # early/north
            np.testing.assert_array_equal(vals[nt:, nlat:, :], ls_vals)  # late/south
            assert (vals[:nt, nlat:, :] == fill).all()  # early/south -> sentinel
            assert (vals[nt:, :nlat, :] == fill).all()  # late/north -> sentinel

    def test_float_fill_reads_back_nan(self, netcdf4_files_factory_2d, local_registry):
        # write the tiles as float32/NaN-fill so the padded band reads back as NaN
        encoding = {"air": {"dtype": "float32", "_FillValue": np.float32("nan")}}
        north, _, south, _ = netcdf4_files_factory_2d(encoding=encoding)
        parser = HDFParser()
        with (
            open_virtual_dataset(
                url=north,
                registry=local_registry,
                parser=parser,
                loadable_variables=LOADABLE_COORDS,
            ) as n,
            open_virtual_dataset(
                url=south,
                registry=local_registry,
                parser=parser,
                loadable_variables=LOADABLE_COORDS,
            ) as s,
        ):
            assert n["air"].dtype == np.dtype("float32")
            union_lat = np.concatenate([n.lat.values, s.lat.values])
            reindexed = n.reindex(lat=union_lat)
            assert isinstance(reindexed["air"].data, ManifestArray)

            nlat = n.sizes["lat"]
            vals = _read_back(
                reindexed["air"].data, ["time", "lat", "lon"], local_registry
            )
            assert np.isfinite(vals[:, :nlat, :]).all()  # real band
            assert np.isnan(vals[:, nlat:, :]).all()  # padded band -> NaN


class TestGeneralWhereStillRejected:
    def test_boolean_masking_raises_clearly(self, array_v3_metadata):
        # we must NOT silently no-op general where(); it should raise, since it
        # would require materializing values.
        metadata = array_v3_metadata(shape=(3,), chunks=(1,), dimension_names=["x"])
        marr = ManifestArray(
            metadata=metadata,
            chunkmanifest=ChunkManifest(
                entries={
                    str(i): {"path": "/a.nc", "offset": i * 4, "length": 4}
                    for i in range(3)
                }
            ),
        )
        ds = xr.Dataset({"foo": ("x", marr)}, coords={"x": [0, 1, 2]})
        with pytest.raises(NotImplementedError, match="materializing|where"):
            ds["foo"].where(ds["x"] > 0).compute()
