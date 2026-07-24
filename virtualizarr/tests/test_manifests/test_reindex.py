"""
Reindex/alignment behaviour exercised through the *public* ManifestArray API.

xarray's reindex/alignment machinery conforms an array to new labels by indexing
it with an integer "gather" indexer that has ``-1`` at every missing position
(e.g. ``[0, 1, 2, -1, -1]``). ManifestArray supports that indexer at the array
level -- ``marr[indexer]`` -- by remapping the chunk grid: a run of ``-1`` that
covers a whole chunk becomes a null-path chunk (reads back as ``fill_value``),
and a chunk-aligned run of real indices keeps its chunk reference. Nothing that
would split a chunk is allowed.

These tests drive that public indexing path directly rather than the private
``chunk_map_from_indexer``/``_reindex_axis`` helpers that implement it.
"""

import numpy as np
import pytest

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import create_v3_array_metadata


def _idx(*vals):
    return np.array(vals, dtype="int64")


def _marr(shape, chunks, entries, dims):
    metadata = create_v3_array_metadata(
        shape=shape,
        chunk_shape=chunks,
        data_type=np.dtype("float32"),
        codecs=[{"configuration": {"endian": "little"}, "name": "bytes"}],
        fill_value=float("nan"),
        dimension_names=dims,
    )
    return ManifestArray(
        metadata=metadata, chunkmanifest=ChunkManifest(entries=entries)
    )


class TestReindexGatherKeepsChunks:
    """An integer gather indexer remaps the chunk grid, lazily, keeping refs."""

    def test_append_null_chunks(self):
        # chunk size 1: every missing (-1) position becomes its own null chunk
        marr = _marr(
            (3,),
            (1,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 4},
                "1": {"path": "/a.nc", "offset": 4, "length": 4},
                "2": {"path": "/a.nc", "offset": 8, "length": 4},
            },
            ["x"],
        )
        result = marr[_idx(0, 1, 2, -1, -1)]

        assert result.shape == (5,)
        assert result.metadata.chunks == (1,)
        # appended positions are absent from the manifest -> null chunks
        assert sorted(result.manifest.dict()) == ["0", "1", "2"]

    def test_chunked_append(self):
        # source chunks [0,1],[2,3]; one all-missing chunk appended -> one null
        marr = _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )
        result = marr[_idx(0, 1, 2, 3, -1, -1)]

        assert result.shape == (6,)
        assert result.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 0, "length": 8},
            "1": {"path": "file:///b.nc", "offset": 8, "length": 8},
        }

    def test_prepend_null_chunks(self):
        marr = _marr(
            (2,),
            (1,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 4},
                "1": {"path": "/a.nc", "offset": 4, "length": 4},
            },
            ["x"],
        )
        result = marr[_idx(-1, -1, 0, 1)]

        assert result.shape == (4,)
        # real chunks shifted to slots 2,3; slots 0,1 are null
        assert sorted(result.manifest.dict()) == ["2", "3"]

    def test_insert_whole_chunk_gap(self):
        # [0,1] [missing] [2,3]: a whole null chunk inserted between two real ones
        marr = _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )
        result = marr[_idx(0, 1, -1, -1, 2, 3)]

        assert result.shape == (6,)
        assert result.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 0, "length": 8},
            "2": {"path": "file:///b.nc", "offset": 8, "length": 8},
        }

    def test_whole_chunk_reorder(self):
        # reversing two whole chunks just swaps their references; no bytes read
        marr = _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )
        result = marr[_idx(2, 3, 0, 1)]

        assert result.manifest.dict() == {
            "0": {"path": "file:///b.nc", "offset": 8, "length": 8},
            "1": {"path": "file:///a.nc", "offset": 0, "length": 8},
        }

    def test_select_trailing_partial_chunk(self):
        # source len 5, chunk 2 -> chunk sizes 2,2,1; selecting just the size-1 tail
        marr = _marr(
            (5,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/a.nc", "offset": 8, "length": 8},
                "2": {"path": "/a.nc", "offset": 16, "length": 4},
            },
            ["x"],
        )
        result = marr[_idx(4)]

        assert result.shape == (1,)
        assert result.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 16, "length": 4},
        }

    def test_only_indexed_axis_remapped_2d(self):
        # a slice on axis 0 + gather on axis 1: only axis 1's grid changes
        marr = _marr(
            (2, 4),
            (2, 2),
            {
                "0.0": {"path": "/a.nc", "offset": 0, "length": 16},
                "0.1": {"path": "/b.nc", "offset": 16, "length": 16},
            },
            ["y", "x"],
        )
        result = marr[:, _idx(0, 1, 2, 3, -1, -1)]

        assert result.shape == (2, 6)
        assert result.metadata.chunks == (2, 2)
        assert result.manifest.dict() == {
            "0.0": {"path": "file:///a.nc", "offset": 0, "length": 16},
            "0.1": {"path": "file:///b.nc", "offset": 16, "length": 16},
        }


class TestReindexGatherRejectsChunkSplits:
    """Anything that would split a chunk raises, naming the offending axis."""

    @pytest.fixture
    def marr(self):
        return _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )

    def test_mixed_present_and_missing_in_a_chunk(self, marr):
        with pytest.raises(NotImplementedError, match="chunk boundaries"):
            marr[_idx(0, 1, 2, -1)]

    def test_sub_chunk_reorder(self, marr):
        with pytest.raises(NotImplementedError, match="chunk boundaries"):
            marr[_idx(1, 0, 2, 3)]

    def test_unaligned_start(self, marr):
        with pytest.raises(NotImplementedError, match="chunk boundaries"):
            marr[_idx(1, 2)]

    def test_partial_target_of_full_source(self, marr):
        # taking only part of source chunk [2,3] would split it
        with pytest.raises(NotImplementedError, match="chunk boundaries"):
            marr[_idx(0, 1, 2)]
