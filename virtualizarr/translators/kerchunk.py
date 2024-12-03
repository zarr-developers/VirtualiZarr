from typing import Any, Mapping, MutableMapping, cast

from xarray import Dataset
from xarray.core.indexes import Index
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import ChunkEntry, ChunkKey
from virtualizarr.readers.common import separate_coords
from virtualizarr.types.kerchunk import (
    KerchunkArrRefs,
    KerchunkStoreRefs,
)
from virtualizarr.zarr import ZArray, ZAttrs, determine_chunk_grid_shape


def virtual_vars_and_metadata_from_kerchunk_refs(
    vds_refs: KerchunkStoreRefs,
    loadable_variables,
    drop_variables,
    virtual_array_class=ManifestArray,
    fs_root: str | None = None,
) -> tuple[Mapping[str, Variable], dict[str, Any], list[str]]:
    """
    Parses all useful information from a set kerchunk references (for a single group).

    Parameters
    ----------
    fs_root
        The root of the fsspec filesystem on which these references were generated.
        Required if any paths are relative in order to turn them into absolute paths (which virtualizarr requires).
    """

    virtual_vars = virtual_vars_from_kerchunk_refs(
        vds_refs,
        drop_variables=drop_variables + loadable_variables,
        virtual_array_class=virtual_array_class,
        fs_root=fs_root,
    )
    ds_attrs = fully_decode_arr_refs(vds_refs["refs"]).get(".zattrs", {})
    coord_names = ds_attrs.pop("coordinates", [])

    return virtual_vars, ds_attrs, coord_names


def extract_group(vds_refs: KerchunkStoreRefs, group: str | None) -> KerchunkStoreRefs:
    """Extract only the part of the kerchunk reference dict that is relevant to a single HDF group"""
    hdf_groups = [
        k.removesuffix(".zgroup") for k in vds_refs["refs"].keys() if ".zgroup" in k
    ]
    if len(hdf_groups) == 1:
        return vds_refs
    else:
        if group is None:
            raise ValueError(
                f"Multiple HDF Groups found. Must specify group= keyword to select one of {hdf_groups}"
            )
        else:
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


def virtual_vars_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: list[str] | None = None,
    virtual_array_class=ManifestArray,
    fs_root: str | None = None,
) -> dict[str, Variable]:
    """
    Translate a store-level kerchunk reference dict into aaset of xarray Variables containing virtualized arrays.

    Parameters
    ----------
    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    var_names = find_var_names(refs)
    if drop_variables is None:
        drop_variables = []
    var_names_to_keep = [
        var_name for var_name in var_names if var_name not in drop_variables
    ]

    vars = {
        var_name: variable_from_kerchunk_refs(
            refs, var_name, virtual_array_class, fs_root=fs_root
        )
        for var_name in var_names_to_keep
    }
    return vars


def dataset_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: list[str] = [],
    virtual_array_class: type = ManifestArray,
    indexes: MutableMapping[str, Index] | None = None,
    fs_root: str | None = None,
) -> Dataset:
    """
    Translate a store-level kerchunk reference dict into an xarray Dataset containing virtualized arrays.

    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    vars = virtual_vars_from_kerchunk_refs(
        refs, drop_variables, virtual_array_class, fs_root=fs_root
    )
    ds_attrs = fully_decode_arr_refs(refs["refs"]).get(".zattrs", {})
    coord_names = ds_attrs.pop("coordinates", [])

    if indexes is None:
        indexes = {}
    data_vars, coords = separate_coords(vars, indexes, coord_names)

    vds = Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=ds_attrs,
    )

    return vds


def variable_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    var_name: str,
    virtual_array_class,
    fs_root: str | None = None,
) -> Variable:
    """Create a single xarray Variable by reading specific keys of a kerchunk references dict."""

    arr_refs = extract_array_refs(refs, var_name)
    chunk_dict, zarray, zattrs = parse_array_refs(arr_refs)
    # we want to remove the _ARRAY_DIMENSIONS from the final variables' .attrs
    dims = zattrs.pop("_ARRAY_DIMENSIONS")
    if chunk_dict:
        manifest = manifest_from_kerchunk_chunk_dict(chunk_dict, fs_root=fs_root)
        varr = virtual_array_class(zarray=zarray, chunkmanifest=manifest)
    elif len(zarray.shape) != 0:
        # empty variables don't have physical chunks, but zarray shows that the variable
        # is at least 1D
        shape = determine_chunk_grid_shape(zarray.shape, zarray.chunks)
        manifest = ChunkManifest(entries={}, shape=shape)
        varr = virtual_array_class(zarray=zarray, chunkmanifest=manifest)
    else:
        # This means we encountered a scalar variable of dimension 0,
        # very likely that it actually has no numeric value and its only purpose
        # is to communicate dataset attributes.
        varr = zarray.fill_value

    return Variable(data=varr, dims=dims, attrs=zattrs)


def manifest_from_kerchunk_chunk_dict(
    kerchunk_chunk_dict: dict[ChunkKey, str | tuple[str] | tuple[str, int, int]],
    fs_root: str | None = None,
) -> ChunkManifest:
    """Create a single ChunkManifest from the mapping of keys to chunk information stored inside kerchunk array refs."""

    chunk_entries: dict[ChunkKey, ChunkEntry] = {}
    for k, v in kerchunk_chunk_dict.items():
        if isinstance(v, (str, bytes)):
            raise NotImplementedError(
                "Reading inlined reference data is currently not supported. [ToDo]"
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
    found_var_names = {key.split("/")[0] for key in refs.keys() if "/" in key}

    return list(found_var_names)


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
) -> tuple[dict, ZArray, ZAttrs]:
    zarray = ZArray.from_kerchunk_refs(arr_refs.pop(".zarray"))
    zattrs = arr_refs.pop(".zattrs", {})
    chunk_dict = arr_refs

    return chunk_dict, zarray, zattrs


def fully_decode_arr_refs(d: dict) -> KerchunkArrRefs:
    """
    Only have to do this because kerchunk.SingleHdf5ToZarr apparently doesn't bother converting .zarray and .zattrs contents to dicts, see https://github.com/fsspec/kerchunk/issues/415 .
    """
    import ujson

    sanitized = d.copy()
    for k, v in d.items():
        if k.startswith("."):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = ujson.loads(v)

    return cast(KerchunkArrRefs, sanitized)
