import pandas as pd
import pytest

from virtualizarr.manifests.reindex import chunk_index_map


class TestChunkIndexMap:
    def test_append_chunk_size_1(self):
        # extend [0,1,2] -> [0,1,2,3,4] with C=1: two new null chunks
        assert chunk_index_map([0, 1, 2], [0, 1, 2, 3, 4], 1) == [0, 1, 2, None, None]

    def test_chunked_append(self):
        # source chunks [0,1],[2,3]; target adds a whole new chunk [4,5] (all missing)
        assert chunk_index_map([0, 1, 2, 3], [0, 1, 2, 3, 4, 5], 2) == [0, 1, None]

    def test_prepend(self):
        # source [2,3] (C=1); target [0,1,2,3] -> two leading null chunks
        assert chunk_index_map([2, 3], [0, 1, 2, 3], 1) == [None, None, 0, 1]

    def test_gap_fill_insert_whole_chunk(self):
        # source chunks [0,1],[4,5]; target [0,1,2,3,4,5] inserts a null chunk in the gap
        assert chunk_index_map([0, 1, 4, 5], [0, 1, 2, 3, 4, 5], 2) == [0, None, 1]

    def test_whole_chunk_reverse_c1(self):
        assert chunk_index_map([0, 1, 2], [2, 1, 0], 1) == [2, 1, 0]

    def test_whole_chunk_reverse_c2(self):
        # source chunks [0,1],[2,3]; target swaps the two chunks
        assert chunk_index_map([0, 1, 2, 3], [2, 3, 0, 1], 2) == [1, 0]

    def test_sort_c1(self):
        # unsorted source [3,1,2] sorted to [1,2,3]
        assert chunk_index_map([3, 1, 2], [1, 2, 3], 1) == [1, 2, 0]

    def test_trailing_partial_source_chunk(self):
        # source [0,1,2,3,4] C=2 -> chunks of size 2,2,1; selecting just the partial tail
        assert chunk_index_map([0, 1, 2, 3, 4], [4], 2) == [2]

    def test_raise_mixed_present_and_missing(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_index_map([0, 1, 2, 3], [0, 1, 2, 5], 2)

    def test_raise_sub_chunk_reorder(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_index_map([0, 1, 2, 3], [1, 0, 2, 3], 2)

    def test_raise_unaligned_start(self):
        with pytest.raises(NotImplementedError, match="split"):
            chunk_index_map([0, 1, 2, 3], [1, 2], 2)

    def test_raise_partial_target_of_full_source(self):
        # taking only half of source chunk [2,3] would split it
        with pytest.raises(NotImplementedError, match="split"):
            chunk_index_map([0, 1, 2, 3], [0, 1, 2], 2)

    def test_raise_non_unique_source(self):
        # pandas refuses get_indexer on a non-unique index
        with pytest.raises(pd.errors.InvalidIndexError):
            chunk_index_map([0, 0, 1], [0, 1], 1)
