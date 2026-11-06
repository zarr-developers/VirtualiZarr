import re
from pathlib import Path
from typing import Any

import numpy as np
from obspec_utils.readers import EagerStoreReader
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.utils import encode_cf_fill_value

# ENVI "data type" codes -> numpy dtype strings.
# See https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html
# Codes not listed here (7, 8, 10, 11, and anything >= 16) are either reserved/unused
# by the ENVI format or specific to file types we don't support (see FILE_TYPE below).
ENVI_DATA_TYPES: dict[int, str] = {
    1: "uint8",
    2: "int16",
    3: "int32",
    4: "float32",
    5: "float64",
    6: "complex64",
    9: "complex128",
    12: "uint16",
    13: "uint32",
    14: "int64",
    15: "uint64",
}

# Maps interleave -> the (dim names, in on-disk axis order).
# ENVI's three interleave schemes lay the same (bands, lines, samples) data out
# differently in the flat binary file; representing the array using the same axis
# order as the file lets a single ManifestArray chunk cover the whole array without
# needing any codec to transpose data.
INTERLEAVE_DIMS: dict[str, tuple[str, str, str]] = {
    "bsq": ("band", "y", "x"),
    "bil": ("y", "band", "x"),
    "bip": ("y", "x", "band"),
}

# The only ENVI "file type" whose data is the flat raw binary array described by
# samples/lines/bands/data type/interleave. Other file types (e.g. "ENVI Spectral
# Library", "TIFF", "HDF", "ENVI RPC") store data in an entirely different format and
# would need a dedicated parser.
SUPPORTED_FILE_TYPE = "envi standard"

# Header keys that are informational only, and can be safely carried through as
# opaque zarr attributes without affecting how the binary array is decoded.
PASSTHROUGH_ATTR_KEYS = (
    "description",
    "sensor type",
    "acquisition time",
    "wavelength",
    "wavelength units",
    "fwhm",
    "bbl",
    "band names",
    "class names",
    "class lookup",
    "classes",
    "security tag",
    "map info",
    "projection info",
    "coordinate system string",
    "x start",
    "y start",
    "default bands",
    "default stretch",
    "z plot titles",
    "z plot range",
    "sun azimuth",
    "sun elevation",
)

_HEADER_LINE_RE = re.compile(r"^(?P<key>[^=]+?)\s*=\s*(?P<value>.*)$")


def _parse_envi_header(text: str) -> dict[str, str]:
    """Parse an ENVI ``.hdr`` file's contents into a dict of lowercase key -> raw value string."""

    lines = text.splitlines()
    if not lines or lines[0].strip().lower() != "envi":
        first_line = lines[0] if lines else ""
        raise ValueError(
            "Not a valid ENVI header file: expected the first line to be 'ENVI', "
            f"but got {first_line!r}"
        )

    fields: dict[str, str] = {}
    i = 1
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            i += 1
            continue

        match = _HEADER_LINE_RE.match(line)
        if match is None:
            i += 1
            continue

        key = match.group("key").strip().lower()
        value = match.group("value").strip()

        # values wrapped in {...} may span multiple lines
        if value.startswith("{") and not value.rstrip().endswith("}"):
            collected = [value]
            i += 1
            while i < n and not lines[i].rstrip().endswith("}"):
                collected.append(lines[i])
                i += 1
            if i < n:
                collected.append(lines[i])
            value = "\n".join(collected)

        fields[key] = value
        i += 1

    return fields


def _parse_list_value(value: str) -> list[str]:
    """Parse a ``{a, b, c}``-style ENVI header value into a list of stripped strings."""

    inner = value.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    items = [item.strip() for item in inner.replace("\n", " ").split(",")]
    return [item for item in items if item != ""]


def _as_attr_value(raw: str) -> str | list[str]:
    """Convert a raw ENVI header value into either a plain string or a list of strings."""

    if raw.strip().startswith("{"):
        return _parse_list_value(raw)
    return raw


def _require_int(fields: dict[str, str], key: str) -> int:
    if key not in fields:
        raise ValueError(
            f"ENVI header is missing required field {key!r}, cannot determine array layout"
        )
    try:
        return int(fields[key])
    except ValueError as e:
        raise ValueError(
            f"ENVI header field {key!r} must be an integer, got {fields[key]!r}"
        ) from e


class ENVIParser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an ENVI raster file.

    ENVI files consist of a flat raw binary data file and a companion plain-text
    ``.hdr`` header file describing its shape, dtype, byte order and interleave
    scheme. This parser supports "ENVI Standard" files (the flat raw binary case) in
    band-sequential (BSQ), band-interleaved-by-line (BIL) and band-interleaved-by-pixel
    (BIP) layouts, with or without whole-file gzip compression.

    Any header option that would require actually rearranging bytes to represent as a
    Zarr array (e.g. a non-"ENVI Standard" ``file type``, an unrecognised ``data type``
    or ``interleave``, or framed/blocked layouts via ``major frame offsets`` /
    ``minor frame offsets``) causes a loud failure rather than silently producing
    incorrect data.

    Parameters
    ----------
    header_url
        The URL of the ``.hdr`` header file, if it cannot be found automatically
        by appending or replacing the data file's extension with ``.hdr``.
    split_chunks_along_outer_axis
        Whether to split the array into one chunk per index along its outermost
        (file-contiguous) axis -- band for BSQ, line for BIL/BIP -- so that reading a
        subset of the array only fetches the relevant bytes. Defaults to ``True``.
        Set to ``False`` to instead represent the whole array as a single chunk.
        Ignored (always a single chunk) for gzip-compressed files, since a gzip
        stream cannot be randomly accessed.
    """

    def __init__(
        self,
        header_url: str | None = None,
        split_chunks_along_outer_axis: bool = True,
    ):
        self.header_url = header_url
        self.split_chunks_along_outer_axis = split_chunks_along_outer_axis

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given ENVI raster file to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input ENVI raw binary data file (e.g., "s3://bucket/file.img").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a Zarr representation of the parsed ENVI file.
        """

        store, path_in_store = registry.resolve(url)

        header_url = self.header_url or _find_header_url(url, store, path_in_store)
        header_store, header_path_in_store = registry.resolve(header_url)
        header_text = (
            EagerStoreReader(store=header_store, path=header_path_in_store)
            .readall()
            .decode("utf-8")
        )
        fields = _parse_envi_header(header_text)

        file_type = fields.get("file type", "ENVI Standard").strip().lower()
        if file_type != SUPPORTED_FILE_TYPE:
            raise NotImplementedError(
                f"Unsupported ENVI 'file type': {fields.get('file type')!r}. "
                f"Only {SUPPORTED_FILE_TYPE!r} (flat raw binary) files are supported by ENVIParser."
            )

        for frame_key in ("major frame offsets", "minor frame offsets"):
            if frame_key in fields:
                raise NotImplementedError(
                    f"ENVI header field {frame_key!r} is present, indicating a framed/blocked "
                    "binary layout (e.g. raw aircraft sensor data) that ENVIParser does not support."
                )

        samples = _require_int(fields, "samples")
        lines = _require_int(fields, "lines")
        bands = _require_int(fields, "bands")

        interleave = fields.get("interleave", "").strip().lower()
        if interleave not in INTERLEAVE_DIMS:
            raise ValueError(
                f"Unsupported ENVI 'interleave' value: {fields.get('interleave')!r}. "
                f"Supported values are {sorted(INTERLEAVE_DIMS)}."
            )
        dim_names = INTERLEAVE_DIMS[interleave]
        dim_sizes = {"band": bands, "y": lines, "x": samples}
        shape = tuple(dim_sizes[dim] for dim in dim_names)

        data_type_code = _require_int(fields, "data type")
        if data_type_code not in ENVI_DATA_TYPES:
            raise ValueError(
                f"Unsupported ENVI 'data type' code: {data_type_code}. "
                f"Supported codes are {sorted(ENVI_DATA_TYPES)}."
            )
        dtype = np.dtype(ENVI_DATA_TYPES[data_type_code])

        byte_order = int(fields.get("byte order", "0"))
        if byte_order not in (0, 1):
            raise ValueError(
                f"Unsupported ENVI 'byte order' value: {byte_order}. Expected 0 (little-endian) or 1 (big-endian)."
            )
        if byte_order == 1:
            dtype = dtype.newbyteorder(">")
        else:
            dtype = dtype.newbyteorder("<")

        header_offset = int(fields.get("header offset", "0"))

        file_compression = int(fields.get("file compression", "0"))
        if file_compression not in (0, 1):
            raise ValueError(
                f"Unsupported ENVI 'file compression' value: {file_compression}. Expected 0 (none) or 1 (gzip)."
            )

        data_file_size = store.head(path_in_store)["size"]
        available_bytes = data_file_size - header_offset
        if available_bytes < 0:
            raise ValueError(
                f"ENVI 'header offset' ({header_offset}) exceeds the data file size ({data_file_size} bytes)."
            )

        expected_bytes = int(np.prod(shape)) * dtype.itemsize
        codecs: list[dict[str, Any]] = []
        if file_compression == 1:
            codecs.append({"name": "gzip", "configuration": {"level": 5}})
        else:
            if expected_bytes != available_bytes:
                raise ValueError(
                    "ENVI data file size does not match the size implied by the header "
                    f"(expected {expected_bytes} bytes of array data after the header offset, "
                    f"but {available_bytes} bytes are available). This usually means some header "
                    "option affecting the binary layout (e.g. a sub-windowed 'x start'/'y start' "
                    "image, or multiple images packed into one file) is not accounted for."
                )

        split_chunks = self.split_chunks_along_outer_axis and file_compression == 0
        if split_chunks:
            n_outer = shape[0]
            chunk_shape = (1,) + shape[1:]
            chunk_nbytes = int(np.prod(chunk_shape)) * dtype.itemsize
            trailing_key = ".".join("0" for _ in shape[1:])
            chunkmanifest = ChunkManifest(
                entries={
                    f"{i}.{trailing_key}": {
                        "path": url,
                        "offset": header_offset + i * chunk_nbytes,
                        "length": chunk_nbytes,
                    }
                    for i in range(n_outer)
                }
            )
        else:
            chunk_shape = shape
            chunk_key = ".".join("0" for _ in shape)
            chunkmanifest = ChunkManifest(
                entries={
                    chunk_key: {
                        "path": url,
                        "offset": header_offset,
                        "length": available_bytes,
                    }
                }
            )

        attrs: dict[str, Any] = {}
        for key in PASSTHROUGH_ATTR_KEYS:
            if key in fields:
                attrs[key.replace(" ", "_")] = _as_attr_value(fields[key])

        if "data ignore value" in fields:
            fill_value = np.array(fields["data ignore value"]).astype(dtype)
            attrs["_FillValue"] = encode_cf_fill_value(fill_value, dtype)

        metadata = create_v3_array_metadata(
            shape=shape,
            data_type=dtype,
            chunk_shape=chunk_shape,
            codecs=codecs,
            dimension_names=dim_names,
            attributes=attrs,
        )
        manifest_array = ManifestArray(metadata=metadata, chunkmanifest=chunkmanifest)

        array_name = Path(path_in_store).stem or "data"
        manifest_group = ManifestGroup(arrays={array_name: manifest_array})

        return ManifestStore(group=manifest_group, registry=registry)


def _find_header_url(url: str, store: Any, path_in_store: str) -> str:
    """Locate the ``.hdr`` header file alongside an ENVI data file.

    Tries appending ``.hdr`` to the full data filename first (e.g. ``foo.img`` ->
    ``foo.img.hdr``), then replacing the data file's extension with ``.hdr`` (e.g.
    ``foo.img`` -> ``foo.hdr``), matching the two conventions used in the wild.
    """

    candidates = [path_in_store + ".hdr"]
    stem_candidate = str(Path(path_in_store).with_suffix(".hdr"))
    if stem_candidate not in candidates:
        candidates.append(stem_candidate)

    for candidate in candidates:
        try:
            store.head(candidate)
        except Exception:
            continue
        # reconstruct a full url for this candidate by swapping the suffix of the data url
        suffix = (
            candidate[len(path_in_store) :]
            if candidate.startswith(path_in_store)
            else None
        )
        if suffix is not None:
            return url + suffix
        return url.rsplit("/", 1)[0] + "/" + Path(candidate).name

    raise FileNotFoundError(
        f"Could not find an ENVI '.hdr' header file for {url!r}. Tried: {candidates}. "
        "Pass header_url=... to ENVIParser explicitly if your header file uses a different naming convention."
    )
