from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
)

import numpy as np
from obspec_utils.protocols import ReadableFile
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
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
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


LINKS_REFERENCES_CONVENTION = {
    "uuid": "93624424-4c95-4eda-b86a-f1880206739f",
    "spec_url": "https://github.com/catalystneuro/zarr-links-references/blob/main/README.md",
    "name": "links_references",
    "description": "Links and object references between nodes of Zarr hierarchies",
}
"""The Links and Object References convention, a draft, used with ``links_and_references=True``."""

REF_CONVENTION = {
    "uuid": "d89b30cf-ed8c-43d5-9a16-b492f0cd8786",
    "schema_url": "https://raw.githubusercontent.com/R-CF/zarr_convention_ref/refs/tags/v2.0.0/schema.json",
    "spec_url": "https://github.com/R-CF/zarr_convention_ref/blob/v2.0.0/README.md",
    "name": "ref",
    "description": "External Reference Convention",
}
"""The ref convention, whose objects identify the targets of links."""

_LINKS_AND_REFERENCES_HINT = (
    "To store object references as the paths of their targets, following the draft Links and "
    "Object References convention, pass HDFParser(links_and_references=True)."
)


def _add_conventions(attrs: dict, *conventions: dict) -> None:
    """Register conventions in a node's zarr_conventions attribute, each once."""
    registered = list(attrs.get("zarr_conventions", []))
    uuids = {entry.get("uuid") for entry in registered if isinstance(entry, dict)}
    registered += [dict(c) for c in conventions if c["uuid"] not in uuids]
    attrs["zarr_conventions"] = registered


def _get_fill_value(dataset: H5Dataset):
    """
    Extract the fill value from an h5py dataset, handling string/bytes dtypes
    that don't return numpy scalars from dataset.fillvalue.
    """
    try:
        raw = dataset.fillvalue
    except RuntimeError:
        return np.ma.default_fill_value(dataset.dtype)
    if h5py.check_vlen_dtype(dataset.dtype) in (str, bytes):
        # Variable-length string fill values come back as raw bytes; the array
        # is virtualized as VariableLengthUTF8, so the fill value must be str.
        # Decode strictly: a fill value that is not valid UTF-8 cannot be
        # represented in that dtype, and silently replacing bytes would corrupt it.
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return raw
    elif isinstance(raw, np.generic):
        return raw.item()
    else:
        return raw


def _construct_manifest_array(
    filepath: str,
    dataset: H5Dataset,
    group: str,
    links_and_references: bool = False,
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
    links_and_references
        Store a dataset of object references as the paths of their targets,
        following the draft Links and Object References convention.

    Returns
    -------
    ManifestArray
    """
    chunks = _chunk_shape(dataset)
    codecs = codecs_from_dataset(dataset)
    attrs = _extract_attrs(dataset, links_and_references)
    dtype = dataset.dtype

    # HDF5 variable-length strings use numpy object dtype, which zarr v3 cannot
    # resolve automatically. Map to StringDType which zarr maps to VariableLengthUTF8.
    # Only remap true variable-length strings: h5py.check_string_dtype also
    # matches fixed-length ("S" kind) dtypes, which zarr handles natively and
    # whose raw chunk bytes a vlen-utf8 codec cannot decode. Both cset flavours
    # count — check_vlen_dtype reports `str` for a utf-8 vlen string dtype and
    # `bytes` for an ascii one.
    # Discriminate against other object-kind HDF5 dtypes (vlen arrays, object/
    # region references) that would silently be coerced to StringDType and
    # produce garbage downstream — those aren't supported yet, so fail loudly.
    # The chunks of a variable-length string dataset hold pointers into the HDF5
    # global heap rather than the strings, so they can't be referenced. Its
    # strings are read at parse time and stored as inlined chunks instead.
    vlen_string = h5py.check_vlen_dtype(dtype) in (str, bytes)
    # An object reference is an address in the HDF5 file, so it can't be
    # referenced either. With links_and_references, each is resolved at parse
    # time to its target's path and stored as an inlined string chunk.
    object_reference = h5py.check_ref_dtype(dtype) is h5py.Reference
    # A compound dataset with a variable-length string or object reference field
    # is read at parse time into a struct with fixed-length string fields.
    compound = None
    if dtype.names is not None and any(dtype[name].kind == "O" for name in dtype.names):
        compound, reference_fields = _read_compound(dataset, links_and_references)
        dtype = compound.dtype
    if vlen_string or (object_reference and links_and_references):
        dtype = np.dtypes.StringDType()
    elif dtype.kind == "O":
        raise NotImplementedError(
            f"HDF5 object dtype {dtype!r} is not a variable-length string and "
            f"is not yet supported by HDFParser. h5py exposes vlen arrays "
            f"(`h5py.vlen_dtype`) and object/region references "
            f"(`h5py.ref_dtype`, `h5py.regionref_dtype`) as numpy object dtype; "
            f"please open an issue if your file needs one of these."
            + (f" {_LINKS_AND_REFERENCES_HINT}" if object_reference else "")
        )

    # Temporarily disable use CF->Codecs - TODO re-enable in subsequent PR.
    # cfcodec = cfcodec_from_dataset(dataset)
    # if cfcodec:
    # codecs.insert(0, cfcodec["codec"])
    # dtype = cfcodec["target_dtype"]
    # attrs.pop("scale_factor", None)
    # attrs.pop("add_offset", None)
    # else:
    # dtype = dataset.dtype

    if "_FillValue" in attrs and dtype.kind not in ("S", "U", "O", "T"):
        encoded_cf_fill_value = encode_cf_fill_value(attrs["_FillValue"], dtype)
        attrs["_FillValue"] = encoded_cf_fill_value

    inlined_strings = vlen_string or (object_reference and links_and_references)
    if inlined_strings:
        # The HDF5 filters compress heap pointers or addresses, not the strings;
        # the inlined chunks are encoded with vlen-utf8 alone, which
        # create_v3_array_metadata adds for a string dtype.
        codec_configs: list[dict] = []
        chunks = _vlen_string_chunk_shape(dataset)
    elif compound is not None:
        # The HDF5 filters compress heap pointers or addresses; the inlined
        # chunks are packed little-endian structs, encoded with bytes alone.
        codec_configs = []
        chunks = _vlen_string_chunk_shape(dataset)
    else:
        codec_configs = [
            zarr_codec_config_to_v3(codec.get_config()) for codec in codecs
        ]

    if object_reference and links_and_references:
        # A null reference is the empty string
        fill_value = ""
        _add_conventions(attrs, LINKS_REFERENCES_CONVENTION)
        attrs["links_references"] = {"references": True}
    elif compound is not None:
        fill_value = None  # zeros and empty strings
        if reference_fields:
            conventions = list(attrs.get("zarr_conventions", []))
            attrs["zarr_conventions"] = [
                *conventions,
                dict(LINKS_REFERENCES_CONVENTION),
            ]
            attrs["links_references"] = {"references": reference_fields}
    else:
        fill_value = _get_fill_value(dataset)
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
    if inlined_strings:
        manifest = _inlined_vlen_string_manifest(
            dataset,
            chunks=chunks,
            fill_value=fill_value,
            read=_reference_paths(dataset) if object_reference else None,
        )
    elif compound is not None:
        manifest = _inlined_array_manifest(compound, chunks=chunks)
    else:
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


def _construct_manifest_group(
    filepath: str,
    reader: ReadableFile,
    *,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
    links_and_references: bool = False,
    _seen: dict | None = None,
) -> ManifestGroup:
    """
    Construct a virtual Group from a HDF dataset.

    With ``links_and_references``, soft and external links, and second names of
    objects with more than one hard link, are recorded in the group's attributes
    following the draft Links and Object References convention, instead of being
    copied.
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
        # The objects already stored, so that a second hard link to one is a link
        seen = {} if _seen is None else _seen
        if links_and_references:
            seen.setdefault(_object_address(g), group_name)
        arrays = {}
        groups = {}
        links = {}
        for key in g.keys():
            if key in drop_variables:
                continue
            if links_and_references:
                link = _link_target(g, key, group_name, filepath, seen)
                if link is not None:
                    links[key] = link
                    continue
            item = g[key]
            if isinstance(item, h5py.Dataset):
                arrays[key] = _construct_manifest_array(
                    filepath, item, group_name, links_and_references
                )
            elif isinstance(item, h5py.Group):
                groups[key] = _construct_manifest_group(
                    filepath,
                    reader,
                    group=str(Path(group) / key) if group is not None else key,
                    links_and_references=links_and_references,
                    _seen=seen,
                )
        attributes = _extract_attrs(g, links_and_references)
        if links:
            _add_conventions(attributes, LINKS_REFERENCES_CONVENTION, REF_CONVENTION)
            attributes["links_references"] = {"links": links}

    return ManifestGroup(arrays=arrays, groups=groups, attributes=attributes)


def _link_target(
    g, key: str, group_name: str, filepath: str, seen: dict
) -> dict | None:
    """
    Return the ref object for a member of a group that is a link, or None for one that is not.

    A soft link's target is relative to the group, as a ref convention node is,
    and may not exist; a relative soft link keeps its path as written. An external link's file is resolved against the file
    being parsed. A member reached by a hard link whose object is already stored
    under another name is a link to that name.
    """
    import posixpath

    import h5py

    link = g.get(key, getlink=True)
    if isinstance(link, h5py.SoftLink):
        if link.path.startswith("/"):
            return {"ref": {"node": posixpath.relpath(link.path, start=group_name)}}
        # HDF5 resolves a relative soft link against its group, as a ref node is
        # resolved against the node that holds it, so the path is kept as written
        return {"ref": {"node": link.path}}
    if isinstance(link, h5py.ExternalLink):
        return {
            "ref": {
                "uri": _resolve_external(filepath, link.filename),
                "node": "/" + link.path.lstrip("/"),
            }
        }
    item = g[key]
    address = _object_address(item)
    if address in seen:
        return {"ref": {"node": posixpath.relpath(seen[address], start=group_name)}}
    seen[address] = item.name
    return None


def _object_address(obj) -> int:
    """
    The address of an object in its file, which is the same for every name of it.

    The file is opened again for each group, and an object's ObjectID differs
    between two openings of a file-like reader, so ObjectIDs can't be compared.
    """
    import h5py

    return h5py.h5o.get_info(obj.id).addr


def _resolve_external(filepath: str, filename: str) -> str:
    """The location of an external link's file: as given if absolute, otherwise next to the file being parsed."""
    import posixpath

    if "://" in filename:
        return filename
    if filename.startswith("/"):
        return f"file://{filename}"
    base, _, _ = filepath.rpartition("/")
    scheme, sep, rest = base.partition("://")
    if not sep:
        return posixpath.normpath(posixpath.join(base, filename))
    root, slash, path = rest.partition("/")
    return (
        f"{scheme}://{root}{slash}{posixpath.normpath(posixpath.join(path, filename))}"
    )


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
        Default is [BlockStoreReader][obspec_utils.readers.BlockStoreReader].
    links_and_references
        Store datasets of HDF5 object references as the paths of their targets,
        following the draft [Links and Object References
        convention](https://github.com/catalystneuro/zarr-links-references): each
        path is relative to the array, a null reference is the empty string, and
        the array is marked with ``"links_references": {"references": true}``.
        Soft and external links, and second names of objects with more than one
        hard link, are recorded in the attributes of the group that holds them
        as ``"links_references": {"links": {name: {"ref": {"node": ...}}}}``,
        with a ``uri`` for an external link, instead of being copied. An
        attribute that holds an object reference, or several, becomes a ref
        convention object, or a list of them, whose path is relative to the
        node that has the attribute.
        See [HDF5 links and object references](../../how_to/hdf5_links_references.md).
        Off by default, while the convention is settled; a dataset of object
        references then raises ``NotImplementedError``, and links are copied.
    """

    def __init__(
        self,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        reader_factory: ReaderFactory = BlockStoreReader,
        links_and_references: bool = False,
    ):
        self.group = group
        self.drop_variables = drop_variables
        self.reader_factory = reader_factory
        self.links_and_references = links_and_references

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
        reader = self.reader_factory(store, path_in_store)
        manifest_group = _construct_manifest_group(
            filepath=url,
            reader=reader,
            group=self.group,
            drop_variables=self.drop_variables,
            links_and_references=self.links_and_references,
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


def _vlen_string_chunk_shape(dataset: H5Dataset) -> tuple[int, ...]:
    """
    Chunk shape for a variable-length string dataset, whose chunks are inlined.

    The HDF5 chunk shape is kept, but trimmed to the array shape: an inlined
    chunk is re-encoded, so unlike a referenced chunk (see ``_chunk_shape``)
    trimming it is always safe.
    """
    shape = dataset.shape
    chunks = dataset.chunks if dataset.chunks is not None else shape
    return tuple(max(min(c, s), 1) for c, s in zip(chunks, shape))


def _inlined_vlen_string_manifest(
    dataset: H5Dataset,
    *,
    chunks: tuple[int, ...],
    fill_value: str | None,
    read: Callable[[tuple[slice, ...]], np.ndarray] | None = None,
) -> ChunkManifest:
    """
    Read an HDF5 variable-length string dataset into inlined chunks.

    Each chunk is read with h5py and encoded the way the zarr ``vlen-utf8`` codec
    encodes it. An edge chunk is padded to the full chunk shape with the fill
    value, as zarr does when writing. A dataset with no allocated storage gets an
    empty manifest, so it reads as the fill value.

    Parameters
    ----------
    dataset
        An h5py variable-length string dataset, of either character set.
    chunks
        The chunk shape, as returned by ``_vlen_string_chunk_shape``.
    fill_value
        The array's fill value.
    read
        Returns the strings of a region of the dataset. Defaults to reading the
        dataset's own strings.

    Returns
    -------
    ChunkManifest
        A manifest in which every chunk is inlined.
    """
    from numcodecs.vlen import VLenUTF8

    shape = dataset.shape
    if dataset.size == 0 or dataset.id.get_storage_size() == 0:
        return ChunkManifest(entries={}, shape=shape)

    codec = VLenUTF8()
    if read is None:
        strings = dataset.asstr()

        def read(region: tuple[slice, ...]) -> np.ndarray:
            return strings[region]

    pad = "" if fill_value is None else fill_value
    grid_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))
    inlined: dict[tuple[int, ...], bytes] = {}
    for index in np.ndindex(grid_shape):
        region = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape)
        )
        values = np.asarray(read(region), dtype=object)
        chunk = np.full(chunks, pad, dtype=object)
        # the trailing Ellipsis makes a 0-d assignment copy the element rather
        # than store the 0-d array itself as an object
        chunk[tuple(slice(0, n) for n in values.shape) + (...,)] = values
        inlined[index] = codec.encode(np.ascontiguousarray(chunk))

    lengths = np.zeros(grid_shape, dtype=np.uint64)
    for index, data in inlined.items():
        lengths[index] = len(data)
    return ChunkManifest.from_arrays(
        paths=np.full(grid_shape, INLINED_CHUNK_PATH, dtype=np.dtypes.StringDType()),  # type: ignore
        offsets=np.zeros(grid_shape, dtype=np.uint64),
        lengths=lengths,
        inlined=inlined,
    )


def _reference_path(dataset: H5Dataset) -> Callable[[object], str]:
    """
    Return a function that turns an object reference held by dataset into a path.

    The path is the target's path relative to the dataset, as a ``ref``
    convention ``node`` within a store is, and a null reference is the empty
    string. Many references usually share a few targets, and the ``name`` of an
    object opened by reference makes HDF5 search the file, so each target's
    path is found once.
    """
    import posixpath

    file = dataset.file
    start = dataset.name
    paths: dict = {}

    def path(reference) -> str:
        if not reference:
            return ""
        target = file[reference]
        if target.id not in paths:
            paths[target.id] = posixpath.relpath(target.name, start=start)
        return paths[target.id]

    return path


def _reference_paths(dataset: H5Dataset) -> Callable[[tuple[slice, ...]], np.ndarray]:
    """Return a function that reads a region of an object reference dataset as paths."""
    path = _reference_path(dataset)

    def read(region: tuple[slice, ...]) -> np.ndarray:
        values = np.asarray(dataset[region], dtype=object)
        return np.vectorize(path, otypes=[object])(values) if values.size else values

    return read


def _read_compound(
    dataset: H5Dataset, links_and_references: bool
) -> tuple[np.ndarray, list[str]]:
    """
    Read a compound dataset that has variable-length string or object reference fields.

    Each such field becomes a fixed-length unicode field, as wide as its longest
    value, and every other field is kept, little-endian. A reference field holds
    the paths of its targets, as an object reference dataset does with
    ``links_and_references``.

    Returns
    -------
    values
        The dataset's values as a packed structured array.
    reference_fields
        The names of the fields that hold object references.
    """
    import h5py

    raw = dataset[()] if dataset.shape == () else dataset[...]
    fields: list[tuple[str, np.dtype]] = []
    columns: dict[str, np.ndarray] = {}
    reference_fields: list[str] = []
    for name in raw.dtype.names:
        field_dtype = raw.dtype[name]
        if field_dtype.kind != "O":
            if field_dtype.names is not None or field_dtype.subdtype is not None:
                columns[name] = np.asarray(raw[name])
                fields.append((name, field_dtype))
            else:
                columns[name] = np.asarray(
                    raw[name], dtype=field_dtype.newbyteorder("<")
                )
                fields.append((name, columns[name].dtype))
            continue
        if h5py.check_vlen_dtype(field_dtype) in (str, bytes):
            values = [
                value.decode("utf-8") if isinstance(value, bytes) else value
                for value in np.asarray(raw[name], dtype=object).ravel()
            ]
        elif h5py.check_ref_dtype(field_dtype) is h5py.Reference:
            if not links_and_references:
                raise NotImplementedError(
                    f"Field {name!r} of compound dataset {dataset.name!r} holds HDF5 "
                    f"object references. {_LINKS_AND_REFERENCES_HINT}"
                )
            path = _reference_path(dataset)
            values = [
                path(value) for value in np.asarray(raw[name], dtype=object).ravel()
            ]
            reference_fields.append(name)
        else:
            raise NotImplementedError(
                f"Field {name!r} of compound dataset {dataset.name!r} has HDF5 object "
                f"dtype {field_dtype!r}, which is not yet supported by HDFParser."
            )
        width = max([len(value) for value in values] + [1])
        columns[name] = np.array(values, dtype=f"<U{width}").reshape(raw.shape)
        fields.append((name, columns[name].dtype))

    packed = np.zeros(raw.shape, dtype=np.dtype(fields))
    for name, column in columns.items():
        packed[name] = column
    return packed, reference_fields


def _inlined_array_manifest(
    values: np.ndarray, *, chunks: tuple[int, ...]
) -> ChunkManifest:
    """
    Store an array read at parse time as inlined chunks, encoded with the bytes codec.

    An edge chunk is padded to the full chunk shape with zeros, as zarr does when
    writing.
    """
    shape = values.shape
    if values.size == 0:
        return ChunkManifest(entries={}, shape=shape)
    grid_shape = tuple(math.ceil(s / c) for s, c in zip(shape, chunks))
    inlined: dict[tuple[int, ...], bytes] = {}
    for index in np.ndindex(grid_shape):
        region = tuple(
            slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(index, chunks, shape)
        )
        chunk = np.zeros(chunks, dtype=values.dtype)
        part = values[region]
        chunk[tuple(slice(0, n) for n in part.shape) + (...,)] = part
        inlined[index] = np.ascontiguousarray(chunk).tobytes()
    lengths = np.zeros(grid_shape, dtype=np.uint64)
    for index, data in inlined.items():
        lengths[index] = len(data)
    return ChunkManifest.from_arrays(
        paths=np.full(grid_shape, INLINED_CHUNK_PATH, dtype=np.dtypes.StringDType()),  # type: ignore
        offsets=np.zeros(grid_shape, dtype=np.uint64),
        lengths=lengths,
        inlined=inlined,
    )


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


def _ref_object(reference, path: Callable) -> dict | None:
    """A ref convention object for an object reference, or None for a null one."""
    return {"ref": {"node": path(reference)}} if reference else None


def _extract_attrs(h5obj: H5Dataset | H5Group, links_and_references: bool = False):
    """
    Extract attributes from an HDF5 group or dataset.

    Parameters
    ----------
    h5obj
        An h5py group or dataset.
    links_and_references
        Write an attribute that holds an object reference, or an array of them,
        as ref convention objects whose node is the target's path relative to
        h5obj; a null reference is None.
    """
    import h5py

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
    path = _reference_path(h5obj) if links_and_references else None

    for n, v in h5obj.attrs.items():
        if n in _HIDDEN_ATTRS:
            continue
        if path is not None:
            if isinstance(v, h5py.Reference) and not isinstance(
                v, h5py.RegionReference
            ):
                attrs[n] = _ref_object(v, path)
                _add_conventions(attrs, REF_CONVENTION)
                continue
            if (
                isinstance(v, np.ndarray)
                and v.dtype.kind == "O"
                and v.size
                and all(isinstance(x, h5py.Reference) for x in v.flat)
            ):
                attrs[n] = np.vectorize(
                    functools.partial(_ref_object, path=path), otypes=[object]
                )(v).tolist()
                _add_conventions(attrs, REF_CONVENTION)
                continue
        if n == "_FillValue":
            v = v
        # Fix some attribute values to avoid JSON encoding exceptions...
        if isinstance(v, bytes):
            v = v.decode("utf-8") or " "
        elif isinstance(v, (np.ndarray, np.number, np.bool_)):
            if v.dtype.kind == "S":
                v = np.strings.decode(np.asarray(v), "utf-8").tolist()
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
        # obj is None for a soft link whose target doesn't exist
        if obj is not None and "_Netcdf4Dimid" in obj.attrs:
            dimension_names.append(name)
    for name, obj in group.items():
        if type(obj) is h5py.Dataset:
            if obj.id.get_storage_size() == 0 and name in dimension_names:
                non_coordinate_dimension_variables.append(name)

    return non_coordinate_dimension_variables
