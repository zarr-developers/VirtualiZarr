from __future__ import annotations

import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
)

import numpy as np
from obspec_utils.protocols import ReadableFile, ReadableStore
from obspec_utils.readers import BlockStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.codecs import zarr_codec_config_to_v3
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.hdf.filters import codecs_from_dataset
from virtualizarr.parsers.typing import ReaderFactory
from virtualizarr.parsers.utils import encode_cf_fill_value
from virtualizarr.types import ChunkKey
from virtualizarr.utils import soft_import

h5py = soft_import("h5py", "reading hdf files", strict=False)


if TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
    from h5py import Group as H5Group


def _construct_manifest_array(
    filepath: str,
    dataset: H5Dataset,
    group: str,
) -> ManifestArray:
    """
    Construct a ManifestArray from an h5py dataset

    Parameters
    ----------
    filepath
        The path of the hdf5 file.
    dataset
        An h5py dataset.
    group
        Name of the group containing this h5py.Dataset.

    Returns
    -------
    ManifestArray
    """
    chunks = _chunk_shape(dataset)
    codecs = codecs_from_dataset(dataset)
    attrs = _extract_attrs(dataset)
    dtype = dataset.dtype

    # Temporarily disable use CF->Codecs - TODO re-enable in subsequent PR.
    # cfcodec = cfcodec_from_dataset(dataset)
    # if cfcodec:
    # codecs.insert(0, cfcodec["codec"])
    # dtype = cfcodec["target_dtype"]
    # attrs.pop("scale_factor", None)
    # attrs.pop("add_offset", None)
    # else:
    # dtype = dataset.dtype

    if "_FillValue" in attrs:
        encoded_cf_fill_value = encode_cf_fill_value(attrs["_FillValue"], dtype)
        attrs["_FillValue"] = encoded_cf_fill_value

    codec_configs = [zarr_codec_config_to_v3(codec.get_config()) for codec in codecs]

    fill_value = dataset.fillvalue.item()
    dims = tuple(_dataset_dims(dataset, group=group))
    metadata = create_v3_array_metadata(
        shape=dataset.shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=fill_value,
        codecs=codec_configs,
        dimension_names=dims,
        attributes=attrs,
    )
    manifest = _dataset_chunk_manifest(filepath, dataset, chunks=chunks)
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _chunk_shape(dataset: H5Dataset) -> tuple[int, ...]:
    """
    Determine the chunk shape to report for an h5py dataset.

    For a dataset along an unlimited (extendable) dimension, h5py reports the
    chunk shape allocated for the full maxshape, which can exceed the actual
    array shape - e.g. a coordinate holding 5 values along an unlimited
    dimension reports ``chunks=(512,)``. An oversized chunk inhibits
    concatenation of the resulting virtual dataset, so trim it down to the array
    shape where it is safe to do so.

    Trimming the chunk shrinks the in-bounds region the chunk covers, so the
    manifest must point at fewer bytes than the full stored chunk. That region
    is only a contiguous byte range - and so expressible as a single manifest
    entry - when the chunk is unfiltered (uncompressed) and only the leading
    (slowest-varying) dimension is trimmed. When an oversized chunk can't be
    trimmed safely (e.g. it is compressed) the original chunk shape is kept: the
    variable still reads correctly (zarr crops the oversized edge chunk) and can
    be written as virtual references, but it can't be concatenated with other
    virtual datasets (the oversized chunk prevents a regular chunk grid). That
    case is surfaced to the user as a warning at
    ``ManifestStore.to_virtual_dataset`` time, suggesting they load the variable
    instead.

    This relies on the same invariant as the sub-chunk slicing in
    ``virtualizarr.manifests.indexing`` (a contiguous sub-range of an
    uncompressed, fixed-order chunk is addressable as a single byte range);
    trimming here is the special case of taking the leading prefix along axis 0.
    """
    shape = dataset.shape
    # Clamp each dim to >= 1: zarr v3 allows shape=(0,) but forbids zero-length
    # chunk dimensions (enforced by zarr-python >= 3.2.0). See
    # https://github.com/zarr-developers/zarr-python/issues/3711.
    if dataset.chunks is None:
        return tuple(max(s, 1) for s in shape)

    chunks = tuple(min(c, max(s, 1)) for c, s in zip(dataset.chunks, shape))
    if chunks == dataset.chunks:
        return chunks

    unfiltered = dataset.id.get_create_plist().get_nfilters() == 0
    leading_dim_only = chunks[1:] == dataset.chunks[1:]
    if unfiltered and leading_dim_only:
        return chunks
    return dataset.chunks


def _resolve_local_path(store: ReadableStore, path_in_store: str) -> str | None:
    """Return the filesystem path of ``path_in_store`` if ``store`` is local.

    When the resolved store is an obstore ``LocalStore`` the file is a real path
    on disk, so we can hand it straight to ``h5py.File`` and let HDF5's own
    index-aware driver walk the chunk index natively - reading each index type
    at native granularity instead of dragging a full block per ~2 KiB index-node
    read through the object-store reader. See the module for why block-based
    reading amplifies the chunk-index walk on chunk-dense files.

    Returns ``None`` for any non-local store, or if the reconstructed path does
    not point at an existing file (in which case we fall back to the reader).
    """
    import obstore.store as obs

    if not isinstance(store, obs.LocalStore):
        return None
    # ``LocalStore.prefix`` may be None (no prefix), a str, or a PosixPath. With
    # no prefix, obstore strips the leading "/" from the path, so the resolved
    # ``path_in_store`` is relative to the filesystem root - rejoin it there.
    prefix = getattr(store, "prefix", None)
    root = str(prefix) if prefix is not None else "/"
    candidate = Path(root) / path_in_store
    # Guard against any reconstruction we didn't anticipate: only take the native
    # path when it actually resolves to a file, otherwise fall back to the reader.
    return str(candidate) if candidate.is_file() else None


def _construct_manifest_group(
    filepath: str,
    reader: ReadableFile | str,
    *,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
) -> ManifestGroup:
    """
    Construct a virtual Group from a HDF dataset.

    ``reader`` is either a file-like reader over the object store or, for local
    files, a filesystem path string that ``h5py.File`` opens with its native
    driver.
    """
    import h5py

    with h5py.File(reader, mode="r") as f:
        if not isinstance(g := f.get(group or "/"), h5py.Group):
            raise ValueError(f"Group {group!r} is not an HDF Group")

        # Several of our test fixtures which use xr.tutorial data have
        # non coord dimensions serialized using big endian dtypes which are not
        # yet supported in zarr-python v3.  We'll drop these variables for the
        # moment until big endian support is included upstream.

        non_coordinate_dimension_vars = _find_non_coord_dimension_vars(group=g)
        drop_variables = set(drop_variables or ()) | set(non_coordinate_dimension_vars)
        group_name = str(g.name)  # NOTE: this will always include leading "/"
        arrays = {
            key: _construct_manifest_array(filepath, dataset, group_name)
            for key in g.keys()
            if key not in drop_variables
            if isinstance(dataset := g[key], h5py.Dataset)
        }
        groups = {
            key: _construct_manifest_group(
                filepath,
                reader,
                group=str(Path(group) / key) if group is not None else key,
            )
            for key in g.keys()
            if key not in drop_variables
            if isinstance(g[key], h5py.Group)
        }
        attributes = _extract_attrs(g)

    return ManifestGroup(arrays=arrays, groups=groups, attributes=attributes)


class HDFParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an HDF5/NetCDF4 file.

    Parameters
    ----------
    group
        Name of the group within the HDF5 file to virtualize.
    drop_variables
        Variables in the file that will be ignored when creating the ManifestStore
        (default: `None`, do not ignore any variables).
    reader_factory
        A callable that creates a file-like reader from a store and path.
        Must return an object implementing the
        [ReadableFile][obspec_utils.protocols.ReadableFile] protocol.
        Defaults to `None`, which uses
        [BlockStoreReader][obspec_utils.readers.BlockStoreReader] for remote
        sources and the native fast path (below) for local files.

        When the source is a local file and ``reader_factory`` is left as
        `None`, the parser bypasses the reader entirely and opens the file with
        h5py's native driver. HDF5 then walks the chunk index at native
        granularity, which for chunk-dense datasets reads orders of magnitude
        fewer bytes than serving each ~2 KiB index-node read through a
        fixed-block reader. Passing an explicit ``reader_factory`` opts back into
        the reader path for local files too.
    """

    def __init__(
        self,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        reader_factory: ReaderFactory | None = None,
    ):
        self.group = group
        self.drop_variables = drop_variables
        # ``None`` means "take the local native fast path when possible, else fall
        # back to BlockStoreReader"; an explicit factory is always honoured.
        self.reader_factory = reader_factory

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given HDF5/NetCDF4 file to produce a VirtualiZarr
        [ManifestStore][virtualizarr.manifests.ManifestStore].

        Parameters
        ----------
        url
            The URL of the input HDF5/NetCDF4 file (e.g., `"s3://bucket/store.zarr"`).
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A [ManifestStore][virtualizarr.manifests.ManifestStore] which provides a Zarr representation of the parsed file.
        """
        store, path_in_store = registry.resolve(url)
        # With no explicit reader_factory, a local file takes the native fast path
        # (h5py opens it directly); everything else goes through the block reader.
        reader: ReadableFile | str
        if self.reader_factory is None:
            reader = _resolve_local_path(store, path_in_store) or BlockStoreReader(
                store, path_in_store
            )
        else:
            reader = self.reader_factory(store, path_in_store)
        manifest_group = _construct_manifest_group(
            filepath=url,
            reader=reader,
            group=self.group,
            drop_variables=self.drop_variables,
        )
        # Convert to a manifest store
        return ManifestStore(registry=registry, group=manifest_group)


def _dataset_chunk_manifest(
    filepath: str,
    dataset: H5Dataset,
    *,
    chunks: tuple[int, ...],
) -> ChunkManifest:
    """
    Generate ChunkManifest for HDF5 dataset.

    Parameters
    ----------
    filepath
        The path of the HDF5 file
    dataset
        h5py dataset for which to create a ChunkManifest
    chunks
        The chunk shape to use, as returned by ``_chunk_shape``. This may be
        smaller than ``dataset.chunks`` when an oversized chunk has been trimmed
        to the array shape (see ``_chunk_shape``), in which case each chunk's
        byte length is recomputed for the trimmed, in-bounds region.

    Returns
    -------
    ChunkManifest
        A Virtualizarr ChunkManifest
    """
    dsid = dataset.id
    if dataset.chunks is None:
        if dsid.get_offset() is None:
            chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
        elif dataset.shape == ():
            chunk_manifest = ChunkManifest.from_arrays(
                paths=np.array(filepath, dtype=np.dtypes.StringDType),  # type: ignore
                offsets=np.array(dsid.get_offset(), dtype=np.uint64),
                lengths=np.array(dsid.get_storage_size(), dtype=np.uint64),
            )
        else:
            key_list = [0] * (len(dataset.shape) or 1)
            key = ".".join(map(str, key_list))

            chunk_entry: ChunkEntry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
                path=filepath, offset=dsid.get_offset(), length=dsid.get_storage_size()
            )
            chunk_key = ChunkKey(key)
            chunk_entries = {chunk_key: chunk_entry}
            chunk_manifest = ChunkManifest(entries=chunk_entries)
    else:
        num_chunks = dsid.get_num_chunks()
        if num_chunks == 0:
            chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
        else:
            grid_shape = tuple(math.ceil(a / b) for a, b in zip(dataset.shape, chunks))
            paths = np.empty(grid_shape, dtype=np.dtypes.StringDType)
            offsets = np.empty(grid_shape, dtype=np.uint64)
            lengths = np.empty(grid_shape, dtype=np.uint64)

            # When an oversized chunk has been trimmed the stored chunk holds
            # more bytes than the in-bounds region, so use the trimmed chunk's
            # byte size (valid because the trimmed region is a contiguous prefix
            # of an unfiltered chunk - see _chunk_shape) rather than blob.size.
            trimmed = chunks != dataset.chunks
            trimmed_length = math.prod(chunks) * dataset.dtype.itemsize

            def get_key(blob):
                return tuple(a // b for a, b in zip(blob.chunk_offset, chunks))

            def add_chunk_info(blob):
                key = get_key(blob)
                paths[key] = filepath
                offsets[key] = blob.byte_offset
                lengths[key] = trimmed_length if trimmed else blob.size

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))
            if has_chunk_iter:
                dsid.chunk_iter(add_chunk_info)
            else:
                for index in range(num_chunks):
                    add_chunk_info(dsid.get_chunk_info(index))

            chunk_manifest = ChunkManifest.from_arrays(
                paths=paths,  # type: ignore
                offsets=offsets,
                lengths=lengths,
            )
    return chunk_manifest


def _dataset_dims(dataset: H5Dataset, group: str = "/") -> list[str]:
    """
    Get a list of dimension scale names attached to input HDF5 dataset.

    This is required by the xarray package to work with Zarr arrays. Only
    one dimension scale per dataset dimension is allowed. If dataset is
    dimension scale, it will be considered as the dimension to itself.

    Parameters
    ----------
    dataset
        An h5py dataset.
    group
        Name of the group we are pulling these dimensions from (default: the root
        group "/"). Required for removing subgroup prefixes.

    Returns
    -------
    list[str]
        List with HDF5 path names of dimension scales attached to input
        dataset.
    """
    import h5py

    dims: list[str] = []

    for n in range(len(dataset.shape)):
        if (num_scales := len(dataset.dims[n])) == 1:
            dims.append(str(dataset.dims[n][0].name))
        elif h5py.h5ds.is_scale(dataset.id):
            dims.append(str(dataset.name))
        elif num_scales > 1:
            raise ValueError(
                f"{dataset.name} has {num_scales} dimension scales attached to "
                f"dimension #{n}; require exactly 1"
            )
        elif num_scales == 0:
            # Some HDF5 files do not have dimension scales.
            # If this is the case, `num_scales` will be 0.
            # In this case, we mimic netCDF4 and assign phony dimension names.
            # See https://github.com/fsspec/kerchunk/issues/41
            dims.append(f"phony_dim_{n}")

    return [dim.removeprefix(group).removeprefix("/") for dim in dims]


def _extract_attrs(h5obj: H5Dataset | H5Group):
    """
    Extract attributes from an HDF5 group or dataset.

    Parameters
    ----------
    h5obj
        An h5py group or dataset.
    """
    _HIDDEN_ATTRS = {
        "REFERENCE_LIST",
        "CLASS",
        "DIMENSION_LIST",
        "NAME",
        "_Netcdf4Dimid",
        "_Netcdf4Coordinates",
        "_nc3_strict",
        "_NCProperties",
    }
    attrs = {}
    for n, v in h5obj.attrs.items():
        if n in _HIDDEN_ATTRS:
            continue
        if n == "_FillValue":
            v = v
        # Fix some attribute values to avoid JSON encoding exceptions...
        if isinstance(v, bytes):
            v = v.decode("utf-8") or " "
        elif isinstance(v, (np.ndarray, np.number, np.bool_)):
            if v.dtype.kind == "S":
                v = v.astype(str)
            elif v.size == 1:
                v = v.flatten()[0]
                if isinstance(v, (np.ndarray, np.number, np.bool_)):
                    v = v.tolist()
            else:
                v = v.tolist()
        elif isinstance(v, h5py._hl.base.Empty):
            v = ""
        if v == "DIMENSION_SCALE":
            continue
        attrs[n] = v
    return attrs


def _find_non_coord_dimension_vars(group: H5Group) -> list[str]:
    import h5py

    dimension_names = []
    non_coordinate_dimension_variables = []
    for name, obj in group.items():
        if "_Netcdf4Dimid" in obj.attrs:
            dimension_names.append(name)
    for name, obj in group.items():
        if type(obj) is h5py.Dataset:
            if obj.id.get_storage_size() == 0 and name in dimension_names:
                non_coordinate_dimension_variables.append(name)

    return non_coordinate_dimension_variables
