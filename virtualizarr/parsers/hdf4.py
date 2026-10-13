from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import ChunkKey
from virtualizarr.parsers.kerchunk.translator import (
    from_kerchunk_refs,
    manifest_from_kerchunk_chunk_dict,
)
from virtualizarr.utils import determine_chunk_grid_shape

# HDF4's magic number, at byte 0 of every HDF4 file.
_MAGIC = b"\x0e\x03\x13\x01"

# NASA products store their global metadata as ODL text blobs under these names.
_ODL_METADATA_PREFIXES = ("CoreMetadata.", "ArchiveMetadata.", "StructMetadata.")

# Keys that kerchunk's decoder mixes into each variable's dict alongside that
# variable's real HDF4 attributes, and which therefore must not become attributes.
_DECODER_KEYS = frozenset({"chunks", "dims", "dtype", "refs"})

# The only compression HDF4 uses in practice, and all that kerchunk decodes; see
# the `comp` table in `kerchunk.hdf4`.
_ZLIB = {"id": "zlib"}


def _decode(url: str, reader_options: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Decode an HDF4 file's tags, returning its variables and global attributes.

    This drives `kerchunk.hdf4.HDF4ToZarr` up to the point where it has finished
    reading the file, and stops there. Everything past that point in kerchunk's
    `translate` serializes the result through a Zarr v2 group in order to emit
    Kerchunk references, which we skip in favour of building ManifestArrays
    directly — that round-trip is pure overhead here, and it is what makes zarr
    reject the zero-length variables HDF4 granules routinely contain (zarr
    requires chunk edges to be positive, and kerchunk uses a variable's
    dimensions as its chunk shape). All of the binary tag decoding still lives in
    kerchunk; we only call it.

    Returns
    -------
    tuple
        The per-variable dicts produced by kerchunk's ``_descend_vg``, keyed by
        variable name, and the file's global attributes.
    """
    # kerchunk (and hence fsspec) is an optional dependency of virtualizarr.
    import fsspec
    from kerchunk.hdf4 import HDF4ToZarr

    decoder = HDF4ToZarr(url, **reader_options)
    with fsspec.open(decoder.path, **(decoder.st or {})) as f:
        decoder.f = f

        if f.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{url} is not an HDF4 file: wrong magic number")

        # The file's data descriptors, held in a linked list of blocks.
        decoder.tags = {}
        while True:
            ddh = decoder.read_ddh()
            for _ in range(ddh["ndd"]):
                ident, info = decoder.read_dd()
                decoder.tags[ident] = info
            if ddh["next"] == 0:
                break
            decoder.f.seek(ddh["next"])

        for tag, ref in decoder.tags:
            decoder._dec(tag, ref)

        attributes = _global_attributes(decoder)
        variables = decoder._descend_vg(*_root_vg(decoder))

    return variables, attributes


def _global_attributes(decoder: Any) -> dict[str, Any]:
    """Parse a file's global attributes out of its "Values" tables."""
    attributes: dict[str, Any] = {}
    for (tag, ref), info in decoder.tags.items():
        if tag != "VH" or info["names"][0].upper() != "VALUES":
            continue
        if not info["name"].startswith(_ODL_METADATA_PREFIXES):
            continue

        table = decoder.tags[("VS", ref)]
        decoder.f.seek(table["offset"])
        # Zero-padded to the table's length.
        blob = decoder.f.read(table["length"]).split(b"\x00", 1)[0]

        # Flatten the ODL text's OBJECT/VALUE pairs into individual attributes.
        name = None
        for line in blob.decode().split("\n"):
            if "OBJECT" in line:
                name = line.split()[-1]
            if "VALUE" in line and name is not None:
                attributes[name] = line.split()[-1].strip('"')
    return attributes


def _root_vg(decoder: Any) -> tuple[str, int]:
    """Identify the root virtual group: the last VG that is not a child of another."""
    roots: set[tuple[str, int]] = set()
    for (tag, ref), info in decoder.tags.items():
        if tag != "VG":
            continue
        for child in zip(info["tag"], info["refs"]):
            if child[0] == "VG":
                roots.discard(child)
        roots.add((tag, ref))
    return sorted(roots, key=lambda vg: vg[1])[-1]


def _manifestarray(
    name: str,
    variable: dict[str, Any],
    url: str,
    fs_root: str,
) -> ManifestArray:
    """Build a ManifestArray from one of kerchunk's decoded variable dicts."""
    shape = tuple(variable["dims"])
    # Only a variable with byte references holds any data; the rest are the
    # zero-length variables HDF4 uses to say "nothing was recorded here".
    refs = variable.get("refs", [])

    zarray = {
        "shape": shape,
        # Kerchunk reports an unchunked variable's dimensions as its chunk shape,
        # which for a zero-length variable means a chunk edge of 0. Zarr requires
        # positive edges, and `from_kerchunk_refs` coerces 0 -> 1 for us.
        "chunks": tuple(variable.get("chunks", shape)),
        "dtype": variable["dtype"],
        # HDF4 records a fill value in its FV tag, which kerchunk does not decode.
        "fill_value": 0,
        "filters": None,
        "compressor": _ZLIB if refs else None,
        "zarr_format": 2,
        # HDF4 dimensions are anonymous, so kerchunk names them after the
        # variable, and gives the zero-length variables a single shared name.
        "dimension_names": [f"{name}_x", f"{name}_y"][: len(shape)] if refs else ["0"],
    }
    attributes = {
        key: value.tolist() if isinstance(value, np.generic) else value
        for key, value in variable.items()
        if key not in _DECODER_KEYS
    }
    metadata = from_kerchunk_refs(zarray, attributes)

    chunk_grid_shape = determine_chunk_grid_shape(metadata.shape, metadata.chunks)
    # Each ref is [chunk key, offset, length] plus, when the chunk is compressed,
    # its compression type — which we have already accounted for above.
    # Kerchunk references are lists, not tuples, despite what the signature of
    # `manifest_from_kerchunk_chunk_dict` says.
    chunk_dict: dict[ChunkKey, Any] = {
        ChunkKey(ref[0]): [url, ref[1], ref[2]] for ref in refs
    }
    if chunk_dict:
        manifest = manifest_from_kerchunk_chunk_dict(
            chunk_dict, fs_root=fs_root, shape=chunk_grid_shape
        )
    else:
        manifest = ChunkManifest(entries={}, shape=chunk_grid_shape)

    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


class HDF4Parser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an HDF4 file.

    Parameters
    ----------
    group
        The group within the file to be used as the Zarr root group for the ManifestStore.
    skip_variables
        Variables in the file that will be ignored when creating the ManifestStore.
    reader_options
        Configuration options used internally for kerchunk's fsspec backend.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        reader_options: dict | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables
        self.reader_options = reader_options or {}

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given HDF4 file to produce a VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input HDF4 file (e.g., "s3://bucket/file.hdf").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the parsed HDF4 file.
        """
        # both group=None and group='' mean to read the root group
        if self.group:
            raise ValueError(
                f'Group "{self.group}" not found: kerchunk flattens an HDF4 file into a '
                "single root group, so only the root group can be read."
            )

        variables, attributes = _decode(url, self.reader_options)
        skip = set(self.skip_variables or ())
        fs_root = Path.cwd().as_uri()

        arrays = {}
        for name, variable in variables.items():
            if not isinstance(variable, dict):
                # A scalar at the root of the file is a global attribute, not a
                # variable. The ODL blobs are parsed into attributes already.
                if not name.startswith(_ODL_METADATA_PREFIXES):
                    attributes[name] = (
                        variable.tolist()
                        if isinstance(variable, np.generic)
                        else variable
                    )
                continue
            if name in skip:
                continue
            if "dims" not in variable:
                raise NotImplementedError(
                    f'Cannot read "{name}": it is a nested HDF4 group, which is not yet supported.'
                )
            arrays[name] = _manifestarray(name, variable, url, fs_root)

        manifestgroup = ManifestGroup(arrays=arrays, attributes=attributes)
        return ManifestStore(group=manifestgroup, registry=registry)
