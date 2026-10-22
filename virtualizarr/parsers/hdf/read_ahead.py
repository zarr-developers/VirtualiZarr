"""Read the chunk indexes of an HDF5 file ahead, in parallel.

h5py finds the chunks of a dataset by walking its chunk index, a B-tree, one
node at a time: it reads a node, then each of its children, and so on. In a
file written in pieces the nodes are spread through the file, so on an object
store each one costs a request, made one after the other. For a dataset with
thousands of chunks that is hundreds of requests in a row, and parsing the
file takes as long as their round trips.

[ChunkIndexReadAhead][virtualizarr.parsers.hdf.read_ahead.ChunkIndexReadAhead]
wraps the reader h5py reads the file through. When h5py reads the root of a
chunk index, the wrapper fetches the rest of the index a level at a time, all
the nodes of a level in one call to the store's ``get_ranges``, and answers
h5py's reads of those nodes from memory. h5py still walks the index itself,
so what it finds is unchanged.

Only version 1 B-trees, which HDF5 writes unless a file is created with
``libver="latest"``, are read ahead. Other chunk indexes are read as before.
"""

from __future__ import annotations

from typing import Any

_SIGNATURE = b"TREE"
_CHUNK_NODE = 1  # node type of a B-tree that indexes the chunks of a dataset
_OFFSET_SIZE = 8
_HEADER_SIZE = 8 + 2 * _OFFSET_SIZE  # signature, type, level, entries used, two siblings
_MAX_DIMENSIONS = 32
# Where a node's addresses are counted from: the start of the file, or the end
# of a userblock before it, whose size is a power of two from 512 bytes
_BASE_ADDRESSES = (0, *(512 << i for i in range(16)))


def parse_chunk_index_node(node: bytes, file_size: int) -> tuple[int, list[int]] | None:
    """
    The level and child addresses of a version 1 B-tree node that indexes chunks.

    Returns None for anything else. A key of the node holds the size and filter
    mask of a chunk and its offset in each dimension of the dataset, plus a
    last offset, for the element, that is zero. The node doesn't say how many
    dimensions the dataset has, so each number is tried until every key ends in
    zero and every child address is in the file. A wrong guess only means
    reading ahead bytes that aren't needed.

    Parameters
    ----------
    node
        The bytes of the node, as h5py reads them.
    file_size
        The size of the file, which every child address must be within.
    """
    if len(node) < _HEADER_SIZE or node[:4] != _SIGNATURE or node[4] != _CHUNK_NODE:
        return None
    level = node[5]
    entries = int.from_bytes(node[6:8], "little")
    if entries == 0:
        return None
    for ndim in range(2, _MAX_DIMENSIONS + 2):
        key_size = 8 + 8 * ndim
        if _HEADER_SIZE + (entries + 1) * key_size + entries * _OFFSET_SIZE > len(node):
            return None
        children = []
        for i in range(entries):
            at = _HEADER_SIZE + i * (key_size + _OFFSET_SIZE)
            if int.from_bytes(node[at + key_size - 8 : at + key_size], "little") != 0:
                break
            child = int.from_bytes(node[at + key_size : at + key_size + _OFFSET_SIZE], "little")
            if not 0 < child < file_size:
                break
            children.append(child)
        else:
            return level, children
    return None


class ChunkIndexReadAhead:
    """
    A file-like reader that reads the chunk indexes of an HDF5 file ahead.

    Reads go to ``reader``, except reads of chunk index nodes that were
    already fetched, which are answered from memory. Everything else, such as
    ``seek`` and ``close``, is passed to ``reader``.

    Parameters
    ----------
    reader
        The reader h5py would otherwise read the file through.
    store
        The store the file is in, which must have ``get_ranges`` and ``head``.
    path
        The path of the file in ``store``.
    """

    def __init__(self, reader: Any, store: Any, path: str) -> None:
        self._reader = reader
        self._store = store
        self._path = path
        self._size: int | None = None
        self._base: int | None = None  # where node addresses are counted from
        self._nodes: dict[int, bytes] = {}  # read ahead, by position in the file
        self._done: set[int] = set()  # nodes whose children are read ahead

    def read(self, size: int = -1, /) -> bytes:
        position = self._reader.tell()
        node = self._nodes.get(position)
        if node is not None and 0 <= size <= len(node):
            self._reader.seek(position + size)
            return node[:size]
        data = self._reader.read(size)
        if data[:4] == _SIGNATURE and position not in self._done:
            self._read_ahead(position, bytes(data))
        return data

    def readinto(self, buffer: Any, /) -> int:
        data = self.read(memoryview(buffer).nbytes)
        memoryview(buffer).cast("B")[: len(data)] = data
        return len(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    def _file_size(self) -> int:
        if self._size is None:
            self._size = int(self._store.head(self._path)["size"])
        return self._size

    def _read_ahead(self, position: int, node: bytes) -> None:
        """Fetch every node under the one at position, a level at a time."""
        self._done.add(position)
        size = len(node)
        level = [node]
        while level:
            wanted: list[int] = []
            for parent in level:
                parsed = parse_chunk_index_node(parent, self._file_size())
                if parsed is not None and parsed[0] > 0:  # a leaf's children are chunks
                    wanted += parsed[1]
            if not wanted:
                return
            base = self._find_base(wanted[0], size)
            if base is None:
                return
            starts = [base + a for a in wanted if base + a not in self._nodes]
            ends = [min(start + size, self._file_size()) for start in starts]
            fetched = self._store.get_ranges(
                self._path, starts=starts, lengths=[e - s for s, e in zip(starts, ends)]
            )
            level = []
            for start, data in zip(starts, fetched):
                data = bytes(data)
                self._nodes[start] = data
                self._done.add(start)
                level.append(data)

    def _find_base(self, address: int, size: int) -> int | None:
        """
        Where node addresses are counted from.

        Addresses in an HDF5 file are counted from its superblock, which a
        userblock puts after the start of the file. Found once, by looking for
        a node at a child's address counted from each place it could be.
        """
        if self._base is None:
            self._base = -1  # none found, so nothing is read ahead
            for base in _BASE_ADDRESSES:
                if base + address + _HEADER_SIZE > self._file_size():
                    break
                (head,) = self._store.get_ranges(
                    self._path, starts=[base + address], lengths=[4]
                )
                if bytes(head) == _SIGNATURE:
                    self._base = base
                    break
        return None if self._base < 0 else self._base
