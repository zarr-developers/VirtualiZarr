import numpy as np
import pytest

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.reindex import chunk_map_from_indexer
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


class TestChunkMapFromIndexer:
    def test_append_chunk_size_1(self):
        assert chunk_map_from_indexer(_idx(0, 1, 2, -1, -1), 1, 3) == [
            0,
            1,
            2,
            None,
            None,
        ]

    def test_chunked_append(self):
        # source chunks [0,1],[2,3]; one new all-missing chunk appended
        assert chunk_map_from_indexer(_idx(0, 1, 2, 3, -1, -1), 2, 4) == [0, 1, None]

    def test_prepend(self):
        assert chunk_map_from_indexer(_idx(-1, -1, 0, 1), 1, 2) == [None, None, 0, 1]

    def test_gap_fill_insert_whole_chunk(self):
        # [0,1] [missing] [2,3]
        assert chunk_map_from_indexer(_idx(0, 1, -1, -1, 2, 3), 2, 4) == [0, None, 1]

    def test_whole_chunk_reverse_c1(self):
        assert chunk_map_from_indexer(_idx(2, 1, 0), 1, 3) == [2, 1, 0]

    def test_whole_chunk_reverse_c2(self):
        assert chunk_map_from_indexer(_idx(2, 3, 0, 1), 2, 4) == [1, 0]

    def test_trailing_partial_source_chunk(self):
        # source len 5, C=2 -> chunk sizes 2,2,1; selecting just the size-1 tail
        assert chunk_map_from_indexer(_idx(4), 2, 5) == [2]

    def test_raise_mixed_present_and_missing(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_map_from_indexer(_idx(0, 1, 2, -1), 2, 4)

    def test_raise_sub_chunk_reorder(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_map_from_indexer(_idx(1, 0, 2, 3), 2, 4)

    def test_raise_unaligned_start(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_map_from_indexer(_idx(1, 2), 2, 4)

    def test_raise_partial_target_of_full_source(self):
        # taking only half of source chunk [2,3] would split it
        with pytest.raises(NotImplementedError, match="split"):
            chunk_map_from_indexer(_idx(0, 1, 2), 2, 4)


class TestReindexAxis:
    """ManifestArray._reindex_axis: the chunk-grid remap the indexer hook drives."""

    def test_pad_with_null_chunk(self):
        marr = _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )
        result = marr._reindex_axis(axis=0, chunk_map=[0, None, 1], new_size=6)

        assert result.shape == (6,)
        assert result.metadata.chunks == (2,)
        assert result.manifest.dict() == {
            "0": {"path": "file:///a.nc", "offset": 0, "length": 8},
            "2": {"path": "file:///b.nc", "offset": 8, "length": 8},
        }

    def test_whole_chunk_reorder(self):
        marr = _marr(
            (4,),
            (2,),
            {
                "0": {"path": "/a.nc", "offset": 0, "length": 8},
                "1": {"path": "/b.nc", "offset": 8, "length": 8},
            },
            ["x"],
        )
        result = marr._reindex_axis(axis=0, chunk_map=[1, 0], new_size=4)

        assert result.manifest.dict() == {
            "0": {"path": "file:///b.nc", "offset": 8, "length": 8},
            "1": {"path": "file:///a.nc", "offset": 0, "length": 8},
        }

    def test_only_target_axis_remapped_2d(self):
        marr = _marr(
            (2, 4),
            (2, 2),
            {
                "0.0": {"path": "/a.nc", "offset": 0, "length": 16},
                "0.1": {"path": "/b.nc", "offset": 16, "length": 16},
            },
            ["y", "x"],
        )
        result = marr._reindex_axis(axis=1, chunk_map=[0, 1, None], new_size=6)

        assert result.shape == (2, 6)
        assert result.metadata.chunks == (2, 2)
        assert result.manifest.dict() == {
            "0.0": {"path": "file:///a.nc", "offset": 0, "length": 16},
            "0.1": {"path": "file:///b.nc", "offset": 16, "length": 16},
        }
