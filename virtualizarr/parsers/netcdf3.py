"""Parser for the netCDF classic (netCDF3) formats: CDF-1, CDF-2 and CDF-5.

Only the file header is read. It is a small, fully-specified big-endian binary
grammar that ends every variable's record with the variable's byte offset into
the file, which is exactly the information a ChunkManifest needs.

See the netCDF Classic Format Specification for CDF-1/CDF-2, and the PnetCDF
CDF-5 specification for the 64-bit-data variant.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from obspec_utils.protocols import ReadableFile
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.utils import encode_cf_fill_value

_MAGIC = b"CDF"

# Tags introducing each of the header's three lists.
_NC_DIMENSION = 10
_NC_VARIABLE = 11
_NC_ATTRIBUTE = 12

# nc_type codes. Everything on disk is big-endian; single-byte types have no
# byte order of their own. Codes 7-11 exist only in CDF-5.
_NC_TYPES: dict[int, np.dtype] = {
    1: np.dtype("i1"),  # NC_BYTE
    2: np.dtype("S1"),  # NC_CHAR
    3: np.dtype(">i2"),  # NC_SHORT
    4: np.dtype(">i4"),  # NC_INT
    5: np.dtype(">f4"),  # NC_FLOAT
    6: np.dtype(">f8"),  # NC_DOUBLE
    7: np.dtype("u1"),  # NC_UBYTE
    8: np.dtype(">u2"),  # NC_USHORT
    9: np.dtype(">u4"),  # NC_UINT
    10: np.dtype(">i8"),  # NC_INT64
    11: np.dtype(">u8"),  # NC_UINT64
}
_CDF5_ONLY_TYPES = frozenset({7, 8, 9, 10, 11})

# A numrecs of all-1-bits means the writer never finalized the record count, so
# it has to be recovered from the file size instead. Read as a signed big-endian
# integer that sentinel is -1, at both the 4- and 8-byte widths.
_STREAMING = -1

# Header fields are padded out to a multiple of this many bytes.
_ALIGN = 4


class _HeaderReader:
    """Sequential big-endian reader over a netCDF3 header.

    CDF-5 widens most count and length fields from 4 to 8 bytes; ``size`` and
    ``offset`` capture which width applies where, so the grammar below reads the
    same for all three versions.
    """

    def __init__(self, f: ReadableFile, version: int):
        self._f = f
        self.version = version
        self._wide = version == 5

    def read(self, n: int) -> bytes:
        buf = self._f.read(n)
        if len(buf) < n:
            raise ValueError(
                f"netCDF3 header is truncated: wanted {n} bytes, got {len(buf)}"
            )
        return buf

    def int32(self) -> int:
        return int.from_bytes(self.read(4), "big", signed=True)

    def int64(self) -> int:
        return int.from_bytes(self.read(8), "big", signed=True)

    def size(self) -> int:
        """A count or length: nelems, dim_length, vsize, and CDF-5 dimids."""
        return self.int64() if self._wide else self.int32()

    def offset(self) -> int:
        """A variable's ``begin``: 4 bytes in CDF-1, 8 bytes in CDF-2 and CDF-5."""
        return self.int32() if self.version == 1 else self.int64()

    def pad(self, nbytes: int) -> None:
        """Consume the padding that follows ``nbytes`` of unaligned content."""
        self.read(-nbytes % _ALIGN)

    def name(self) -> str:
        nelems = self.size()
        raw = self.read(nelems)
        self.pad(nelems)
        return raw.decode("utf-8")

    def values(self, nc_type: int, nelems: int) -> Any:
        """Read an attribute's values, as JSON-serializable Python objects."""
        dtype = _NC_TYPES[nc_type]
        nbytes = dtype.itemsize * nelems
        raw = self.read(nbytes)
        self.pad(nbytes)
        if nc_type == 2:  # NC_CHAR attributes are text
            return raw.decode("utf-8", errors="replace").rstrip("\x00")
        values = np.frombuffer(raw, dtype)
        # A length-1 attribute is scalar-valued by convention, matching netcdf-c.
        return values[0].item() if nelems == 1 else values.tolist()

    def att_list(self) -> dict[str, Any]:
        tag = self.int32()
        nelems = self.size()
        if tag == 0:
            if nelems != 0:
                raise ValueError(f"expected an empty attribute list, got {nelems}")
            return {}
        if tag != _NC_ATTRIBUTE:
            raise ValueError(f"expected an attribute list, got tag {tag}")
        attrs = {}
        for _ in range(nelems):
            name = self.name()
            nc_type = self.int32()
            self._check_type(nc_type)
            attrs[name] = self.values(nc_type, self.size())
        return attrs

    def _check_type(self, nc_type: int) -> None:
        if nc_type not in _NC_TYPES:
            raise ValueError(f"unknown netCDF3 type code {nc_type}")
        if nc_type in _CDF5_ONLY_TYPES and self.version != 5:
            raise ValueError(
                f"type code {nc_type} is only valid in CDF-5, but this file is "
                f"CDF-{self.version}"
            )


@dataclass
class _Variable:
    name: str
    dtype: np.dtype
    # Shape as declared, with 0 in the first position for record variables.
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    attributes: dict[str, Any]
    # Bytes allocated per record (record variables) or in total, including padding.
    vsize: int
    # File offset of the variable's data, or of its slice within the first record.
    begin: int
    is_record: bool


@dataclass
class _Header:
    version: int
    numrecs: int
    dimensions: dict[str, int] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    variables: list[_Variable] = field(default_factory=list)
    # Stride between consecutive records; see _record_size.
    recsize: int = 0


def _parse_header(f: ReadableFile) -> _Header:
    """Parse a netCDF3 header from the start of ``f``."""
    magic = f.read(4)
    if magic[:3] != _MAGIC:
        raise ValueError(
            f"not a netCDF3 file: expected magic {_MAGIC!r}, got {magic!r}"
        )
    version = magic[3]
    if version not in (1, 2, 5):
        raise ValueError(
            f"unknown netCDF classic format version {version}; expected CDF-1, CDF-2 or CDF-5"
        )

    r = _HeaderReader(f, version)
    numrecs = r.size()

    header = _Header(version=version, numrecs=numrecs)

    # dim_list
    tag = r.int32()
    nelems = r.size()
    if tag == _NC_DIMENSION:
        for _ in range(nelems):
            name = r.name()
            header.dimensions[name] = r.size()
    elif tag != 0 or nelems != 0:
        raise ValueError(f"expected a dimension list, got tag {tag}")

    header.attributes = r.att_list()

    # var_list
    dim_names = list(header.dimensions)
    tag = r.int32()
    nelems = r.size()
    if tag == _NC_VARIABLE:
        for _ in range(nelems):
            name = r.name()
            ndims = r.size()
            # CDF-5 widens dimids to 8 bytes along with the other count fields.
            dimids = [r.size() for _ in range(ndims)]
            attributes = r.att_list()
            nc_type = r.int32()
            r._check_type(nc_type)
            vsize = r.size()
            begin = r.offset()

            dimensions = tuple(dim_names[d] for d in dimids)
            shape = tuple(header.dimensions[d] for d in dimensions)
            # The unlimited dimension is the one recorded with length 0, and it
            # can only ever be a variable's first dimension.
            is_record = bool(shape) and shape[0] == 0
            header.variables.append(
                _Variable(
                    name=name,
                    dtype=_NC_TYPES[nc_type],
                    shape=shape,
                    dimensions=dimensions,
                    attributes=attributes,
                    vsize=vsize,
                    begin=begin,
                    is_record=is_record,
                )
            )
    elif tag != 0 or nelems != 0:
        raise ValueError(f"expected a variable list, got tag {tag}")

    header.recsize = _record_size(header.variables)
    return header


def _resolve_streaming_numrecs(header: _Header, file_size: int) -> int:
    """Recover the record count of a file whose header never had it written."""
    record_vars = [v for v in header.variables if v.is_record]
    if not record_vars or header.recsize == 0:
        return 0
    # Records start at the first record variable's slice in record 0.
    start = min(v.begin for v in record_vars)
    return max(0, (file_size - start) // header.recsize)


def _inner_shape(var: _Variable) -> tuple[int, ...]:
    """The shape of one record's slice, for a record variable."""
    return var.shape[1:]


def _record_nbytes(var: _Variable) -> int:
    """The unpadded size of one record's slice of a record variable."""
    return math.prod(_inner_shape(var)) * var.dtype.itemsize


def _record_size(variables: list[_Variable]) -> int:
    """The stride between consecutive records.

    Normally each record variable contributes its ``vsize``, which the writer
    already rounded up to a 4-byte boundary. The exception is a file holding
    exactly one record variable: its slices are written back to back with no
    padding between them, even though ``vsize`` in the header is still rounded
    up. Using ``vsize`` there would over-stride and read the wrong bytes.
    """
    record_vars = [v for v in variables if v.is_record]
    if len(record_vars) == 1:
        return _record_nbytes(record_vars[0])
    return sum(v.vsize for v in record_vars)


def _build_manifest_array(var: _Variable, header: _Header, url: str) -> ManifestArray:
    """Build a ManifestArray for one netCDF3 variable.

    Non-record variables are contiguous, so they become a single chunk. Record
    variables are interleaved one record at a time, so they become one chunk per
    record, strided by the file's record size.
    """
    attributes = dict(var.attributes)
    # The zarr fill value is the native scalar, but the CF attribute has to carry
    # the encoded form that xarray's zarr backend expects (base64 for floats).
    fill_value = attributes.get("_FillValue")
    if fill_value is not None and var.dtype.kind not in ("S", "U", "O", "T"):
        attributes["_FillValue"] = encode_cf_fill_value(fill_value, var.dtype)

    if var.is_record:
        shape = (header.numrecs,) + _inner_shape(var)
        # One record per chunk: the natural chunking of the on-disk layout.
        chunk_shape = (1,) + _inner_shape(var)
    else:
        shape = var.shape
        chunk_shape = var.shape

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=var.dtype,
        # Zarr forbids a zero-length chunk edge, so a variable with a zero-length
        # dimension keeps a positive chunk shape over an empty chunk grid.
        chunk_shape=tuple(max(c, 1) for c in chunk_shape),
        fill_value=fill_value,
        attributes=attributes,
        dimension_names=var.dimensions,
    )

    if var.is_record:
        nbytes = _record_nbytes(var)
        # Chunk grid is one cell per record, and a single cell across every
        # other axis, since each chunk spans that axis in full.
        grid_shape = (header.numrecs,) + (1,) * len(_inner_shape(var))
        offsets = var.begin + np.arange(header.numrecs, dtype=np.uint64) * np.uint64(
            header.recsize
        )
        manifest = ChunkManifest.from_arrays(
            paths=np.full(grid_shape, url, dtype=np.dtypes.StringDType()),  # type: ignore[arg-type]
            offsets=offsets.reshape(grid_shape).astype(np.uint64),
            lengths=np.full(grid_shape, nbytes, dtype=np.uint64),
            validate_paths=False,
        )
    elif 0 in var.shape:
        # No data was ever written, so there are no chunks to point at.
        manifest = ChunkManifest(entries={}, shape=(0,) * len(var.shape))
    else:
        nbytes = math.prod(var.shape) * var.dtype.itemsize
        key = ".".join(["0"] * len(var.shape)) or "0"
        manifest = ChunkManifest(
            entries={key: {"path": url, "offset": var.begin, "length": nbytes}},
            shape=(1,) * len(var.shape),
        )

    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


class NetCDF3Parser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from a netCDF3 file.

    Supports all three netCDF classic formats: CDF-1 (classic), CDF-2 (64-bit
    offset) and CDF-5 (64-bit data).

    Parameters
    ----------
    group
        The group within the file to be used as the Zarr root group for the ManifestStore.
        netCDF3 files are flat, so only the root group exists.
    skip_variables
        Variables in the file that will be ignored when creating the ManifestStore.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given netCDF3 file to produce a VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input netCDF3 file (e.g., "s3://bucket/file.nc").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the parsed netCDF3 file.
        """
        if self.group not in (None, "", "/"):
            raise ValueError(
                f'netCDF3 files contain only a root group, so group="{self.group}" cannot be opened'
            )

        store, path_in_store = registry.resolve(url)
        # The header sits at the front of the file, so a block-buffered reader
        # fetches it without reading the data that follows.
        reader = BlockStoreReader(store=store, path=path_in_store)
        header = _parse_header(reader)

        if header.numrecs == _STREAMING:
            file_size = reader.seek(0, 2)
            header.numrecs = _resolve_streaming_numrecs(header, file_size)

        skip = set(self.skip_variables or ())
        arrays = {
            var.name: _build_manifest_array(var, header, url)
            for var in header.variables
            if var.name not in skip
        }

        manifest_group = ManifestGroup(arrays=arrays, attributes=header.attributes)
        return ManifestStore(group=manifest_group, registry=registry)
