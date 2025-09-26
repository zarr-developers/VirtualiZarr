from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import ujson
from zarr.core.common import JSON
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr.codecs import (
    zarr_codec_config_to_v3,
)
from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
)
from virtualizarr.manifests.manifest import ChunkEntry, ChunkKey
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.types.kerchunk import (
    KerchunkArrRefs,
    KerchunkStoreRefs,
)
from virtualizarr.utils import determine_chunk_grid_shape


def from_kerchunk_refs(decoded_arr_refs_zarray, zattrs) -> "ArrayV3Metadata":
    """
    Convert a decoded zarr array (.zarray) reference to an ArrayV3Metadata object.
    This function processes the given decoded Zarr array reference dictionary,
    to construct and return an ArrayV3Metadata object based on the provided information.

    Parameters
    ----------
    decoded_arr_refs_zarray
        A dictionary containing the decoded Zarr array reference information.
        Expected keys include "dtype", "fill_value", "zarr_format", "filters",
        "compressor", "chunks", and "shape".

    Returns
    -------
    ArrayV3Metadata

    Raises
    ------
    ValueError
        If the Zarr format specified in the input dictionary is not 2 or 3.
    """
    # coerce type of fill_value as kerchunk can be inconsistent with this
    dtype = np.dtype(decoded_arr_refs_zarray["dtype"])
    fill_value = decoded_arr_refs_zarray["fill_value"]
    if np.issubdtype(dtype, np.floating) and (
        fill_value is None or fill_value == "NaN" or fill_value == "nan"
    ):
        fill_value = np.nan

    zarr_format = int(decoded_arr_refs_zarray["zarr_format"])
    if zarr_format not in (2, 3):
        raise ValueError(f"Zarr format must be 2 or 3, but got {zarr_format}")
    filters = (
        decoded_arr_refs_zarray.get("filters", []) or []
    )  # Ensure filters is a list
    compressor = decoded_arr_refs_zarray.get("compressor")  # Might be None

    # Ensure compressor is a list before unpacking
    codec_configs = [*filters, *(compressor if compressor is not None else [])]
    numcodec_configs = [zarr_codec_config_to_v3(config) for config in codec_configs]
    dimension_names = decoded_arr_refs_zarray["dimension_names"]
    return create_v3_array_metadata(
        chunk_shape=tuple(decoded_arr_refs_zarray["chunks"]),
        data_type=dtype,
        codecs=numcodec_configs,
        fill_value=fill_value,
        shape=tuple(decoded_arr_refs_zarray["shape"]),
        dimension_names=dimension_names,
        attributes=zattrs,
    )


def manifestgroup_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    group: str | None = None,
    fs_root: str | None = None,
    skip_variables: Iterable[str] | None = None,
) -> ManifestGroup:
    """
    Construct a ManifestGroup from a dictionary of kerchunk references.

    Parameters
    ----------
    refs
        The Kerchunk references, as a dictionary.
    group
        Default is to build a store from the root group.
    fs_root
        The root of the fsspec filesystem on which these references were generated.
        Required if any paths are relative in order to turn them into absolute paths (which virtualizarr requires).
    skip_variables
        Variables to ignore when creating the ManifestGroup.

    Returns
    -------
    ManifestGroup
        ManifestGroup representation of the virtual chunk references.
    """
    # both group=None and group='' mean to read root group
    if group:
        refs = extract_group(refs, group)

    arr_names = find_var_names(refs)
    if skip_variables:
        arr_names = [var for var in arr_names if var not in skip_variables]

    # TODO support iterating over multiple nested groups
    marrs = {
        arr_name: manifestarray_from_kerchunk_refs(refs, arr_name, fs_root=fs_root)
        for arr_name in arr_names
    }

    # TODO probably need to parse the group-level attributes more here
    attributes = fully_decode_arr_refs(refs["refs"]).get(".zattrs", {})

    manifestgroup = ManifestGroup(arrays=marrs, attributes=attributes)
    return manifestgroup


def extract_group(vds_refs: KerchunkStoreRefs, group: str) -> KerchunkStoreRefs:
    """
    Extract only the part of the kerchunk reference dict that is relevant to a single HDF group.

    Parameters
    ----------
    vds_refs
    group
        Should be a non-empty string
    """
    hdf_groups = [
        k.removesuffix(".zgroup") for k in vds_refs["refs"].keys() if ".zgroup" in k
    ]

    # Ensure supplied group kwarg is consistent with kerchunk keys
    if not group.endswith("/"):
        group += "/"
    if group.startswith("/"):
        group = group.removeprefix("/")

    if group not in hdf_groups:
        raise ValueError(f'Group "{group}" not found in {hdf_groups}')

    # Filter by group prefix and remove prefix from all keys
    groupdict = {
        k.removeprefix(group): v
        for k, v in vds_refs["refs"].items()
        if k.startswith(group)
    }
    # Also remove group prefix from _ARRAY_DIMENSIONS
    for k, v in groupdict.items():
        if isinstance(v, str):
            groupdict[k] = v.replace("\\/", "/").replace(group, "")

    vds_refs["refs"] = groupdict

    return KerchunkStoreRefs(vds_refs)


def manifestarray_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    var_name: str,
    fs_root: str | None = None,
) -> ManifestArray:
    """Create a single ManifestArray by reading specific keys of a kerchunk references dict."""

    arr_refs = extract_array_refs(refs, var_name)

    # TODO probably need to update internals of this to use ArrayV3Metadata more neatly
    chunk_dict, metadata, zattrs = parse_array_refs(arr_refs)
    # we want to remove the _ARRAY_DIMENSIONS from the final variables' .attrs
    if chunk_dict:
        manifest = manifest_from_kerchunk_chunk_dict(chunk_dict, fs_root=fs_root)
        marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    elif len(metadata.shape) != 0:
        # empty variables don't have physical chunks, but zarray shows that the variable
        # is at least 1D

        shape = determine_chunk_grid_shape(
            metadata.shape,
            metadata.chunks,
        )
        manifest = ChunkManifest(entries={}, shape=shape)
        marr = ManifestArray(metadata=metadata, chunkmanifest=manifest)
    else:
        # This means we encountered a scalar variable of dimension 0,
        # very likely that it actually has no numeric value and its only purpose
        # is to communicate dataset attributes.
        marr = metadata.fill_value

    return marr


def manifest_from_kerchunk_chunk_dict(
    kerchunk_chunk_dict: dict[ChunkKey, str | tuple[str] | tuple[str, int, int]],
    fs_root: str | None = None,
) -> ChunkManifest:
    """Create a single ChunkManifest from the mapping of keys to chunk information stored inside kerchunk array refs."""

    chunk_entries: dict[ChunkKey, ChunkEntry] = {}
    for k, v in kerchunk_chunk_dict.items():
        if isinstance(v, (str, bytes)):
            raise NotImplementedError(
                "Reading inlined reference data is currently not supported."
                "See https://github.com/zarr-developers/VirtualiZarr/issues/489",
            )
        elif not isinstance(v, (tuple, list)):
            raise TypeError(f"Unexpected type {type(v)} for chunk value: {v}")
        chunk_entries[k] = chunkentry_from_kerchunk(v, fs_root=fs_root)
    return ChunkManifest(entries=chunk_entries)


def chunkentry_from_kerchunk(
    path_and_byte_range_info: tuple[str] | tuple[str, int, int],
    fs_root: str | None = None,
) -> ChunkEntry:
    """Create a single validated ChunkEntry object from whatever kerchunk contains under that chunk key."""
    from upath import UPath

    if len(path_and_byte_range_info) == 1:
        path = path_and_byte_range_info[0]
        offset = 0
        length = UPath(path).stat().st_size
    else:
        path, offset, length = path_and_byte_range_info
    return ChunkEntry.with_validation(  # type: ignore[attr-defined]
        path=path, offset=offset, length=length, fs_root=fs_root
    )


def find_var_names(ds_reference_dict: KerchunkStoreRefs) -> list[str]:
    """Find the names of zarr variables in this store/group."""

    refs = ds_reference_dict["refs"]

    found_var_names = []
    for key in refs.keys():
        # has to capture "foo/.zarray", but ignore ".zgroup", ".zattrs", and "subgroup/bar/.zarray"
        # TODO this might be a sign that we shoulzd introduce a KerchunkGroupRefs type and cut down the references before getting to this point...
        if key not in (".zgroup", ".zattrs", ".zmetadata"):
            first_part, second_part, *_ = key.split("/")
            if second_part == ".zarray":
                found_var_names.append(first_part)

    return found_var_names


def extract_array_refs(
    ds_reference_dict: KerchunkStoreRefs, var_name: str
) -> KerchunkArrRefs:
    """Extract only the part of the kerchunk reference dict that is relevant to this one zarr array"""

    found_var_names = find_var_names(ds_reference_dict)

    refs = ds_reference_dict["refs"]
    if var_name in found_var_names:
        # TODO these function probably have more loops in them than they need to...

        arr_refs = {
            key.split("/")[1]: refs[key]
            for key in refs.keys()
            if var_name == key.split("/")[0]
        }

        return fully_decode_arr_refs(arr_refs)

    else:
        raise KeyError(
            f"Could not find zarr array variable name {var_name}, only {found_var_names}"
        )


def parse_array_refs(
    arr_refs: KerchunkArrRefs,
) -> tuple[dict, ArrayV3Metadata, dict[str, JSON]]:
    zattrs = arr_refs.pop(".zattrs", {})
    dims = zattrs.pop("_ARRAY_DIMENSIONS")
    zarray = arr_refs.pop(".zarray")
    zarray["dimension_names"] = dims
    metadata = from_kerchunk_refs(zarray, zattrs)
    chunk_dict = arr_refs

    return chunk_dict, metadata, zattrs


def fully_decode_arr_refs(d: dict) -> KerchunkArrRefs:
    """
    Only have to do this because kerchunk.SingleHdf5ToZarr apparently doesn't bother converting .zarray and .zattrs contents to dicts, see https://github.com/fsspec/kerchunk/issues/415 .
    """

    sanitized = d.copy()
    for k, v in d.items():
        if k.startswith("."):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = ujson.loads(v)

    return cast(KerchunkArrRefs, sanitized)
