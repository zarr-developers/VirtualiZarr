import warnings
from pathlib import Path
from typing import Any, MutableMapping, Optional, cast

import ujson  # type: ignore
from xarray import Dataset
from xarray.core.indexes import Index
from xarray.core.variable import Variable

from virtualizarr.backend import FileType, separate_coords
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.types.kerchunk import (
    KerchunkArrRefs,
    KerchunkStoreRefs,
)
from virtualizarr.utils import _fsspec_openfile_from_filepath
from virtualizarr.zarr import ZArray, ZAttrs


# TODO shouldn't this live in backend.py? Because it's not just useful for the kerchunk-specific readers...
def _automatically_determine_filetype(
    *,
    filepath: str,
    reader_options: Optional[dict[str, Any]] = {},
) -> FileType:
    if Path(filepath).suffix == ".zarr":
        # TODO we could imagine opening an existing zarr store, concatenating it, and writing a new virtual one...
        raise NotImplementedError()

    # Read magic bytes from local or remote file
    fpath = _fsspec_openfile_from_filepath(
        filepath=filepath, reader_options=reader_options
    )
    magic_bytes = fpath.read(8)
    fpath.close()

    if magic_bytes.startswith(b"CDF"):
        filetype = FileType.netcdf3
    elif magic_bytes.startswith(b"\x0e\x03\x13\x01"):
        raise NotImplementedError("HDF4 formatted files not supported")
    elif magic_bytes.startswith(b"\x89HDF"):
        filetype = FileType.hdf5
    elif magic_bytes.startswith(b"GRIB"):
        filetype = FileType.grib
    elif magic_bytes.startswith(b"II*"):
        filetype = FileType.tiff
    elif magic_bytes.startswith(b"SIMPLE"):
        filetype = FileType.fits
    else:
        raise NotImplementedError(
            f"Unrecognised file based on header bytes: {magic_bytes}"
        )

    return filetype


def read_kerchunk_references_from_file(
    filepath: str,
    filetype: FileType | None,
    group: str | None,
    reader_options: Optional[dict[str, Any]] = None,
) -> KerchunkStoreRefs:
    """
    Read a single legacy file and return kerchunk references to its contents.

    Parameters
    ----------
    filepath : str, default: None
        File path to open as a set of virtualized zarr arrays.
    filetype : FileType, default: None
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        If not provided will attempt to automatically infer the correct filetype from the the filepath's extension.
    group : str, default is None
        Path to the HDF5/netCDF4 group in the given file to open. Given as a str, supported by filetypes “netcdf4” and “hdf5”.
        Dict passed into Kerchunk file readers. Note: Each Kerchunk file reader has distinct arguments,
        so ensure reader_options match selected Kerchunk reader arguments.
    """

    if reader_options is None:
        reader_options = {}

    if filetype is None:
        filetype = _automatically_determine_filetype(
            filepath=filepath, reader_options=reader_options
        )

    # if filetype is user defined, convert to FileType
    filetype = FileType(filetype)

    if filetype.name.lower() == "netcdf3":
        from kerchunk.netCDF3 import NetCDF3ToZarr

        refs = NetCDF3ToZarr(filepath, inline_threshold=0, **reader_options).translate()

    elif filetype.name.lower() == "hdf5" or filetype.name.lower() == "netcdf4":
        from kerchunk.hdf import SingleHdf5ToZarr

        refs = SingleHdf5ToZarr(
            filepath, inline_threshold=0, **reader_options
        ).translate()

        refs = extract_group(refs, group)

    elif filetype.name.lower() == "grib":
        # TODO Grib files should be handled as a DataTree object
        # see https://github.com/TomNicholas/VirtualiZarr/issues/11
        raise NotImplementedError(f"Unsupported file type: {filetype}")
    elif filetype.name.lower() == "tiff":
        from kerchunk.tiff import tiff_to_zarr

        reader_options.pop("storage_options", {})
        warnings.warn(
            "storage_options have been dropped from reader_options as they are not supported by kerchunk.tiff.tiff_to_zarr",
            UserWarning,
        )

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = {"refs": tiff_to_zarr(filepath, **reader_options)}
    elif filetype.name.lower() == "fits":
        from kerchunk.fits import process_file

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = {"refs": process_file(filepath, **reader_options)}
    else:
        raise NotImplementedError(f"Unsupported file type: {filetype.name}")

    # TODO validate the references that were read before returning?
    return refs


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
        var_name: variable_from_kerchunk_refs(refs, var_name, virtual_array_class)
        for var_name in var_names_to_keep
    }
    return vars


def dataset_from_kerchunk_refs(
    refs: KerchunkStoreRefs,
    drop_variables: list[str] = [],
    virtual_array_class: type = ManifestArray,
    indexes: MutableMapping[str, Index] | None = None,
) -> Dataset:
    """
    Translate a store-level kerchunk reference dict into an xarray Dataset containing virtualized arrays.

    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    """

    vars = virtual_vars_from_kerchunk_refs(refs, drop_variables, virtual_array_class)
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
    refs: KerchunkStoreRefs, var_name: str, virtual_array_class
) -> Variable:
    """Create a single xarray Variable by reading specific keys of a kerchunk references dict."""

    arr_refs = extract_array_refs(refs, var_name)
    chunk_dict, zarray, zattrs = parse_array_refs(arr_refs)
    # we want to remove the _ARRAY_DIMENSIONS from the final variables' .attrs
    dims = zattrs.pop("_ARRAY_DIMENSIONS")
    if chunk_dict:
        manifest = ChunkManifest._from_kerchunk_chunk_dict(chunk_dict)
        varr = virtual_array_class(zarray=zarray, chunkmanifest=manifest)
    else:
        # This means we encountered a scalar variable of dimension 0,
        # very likely that it actually has no numeric value and its only purpose
        # is to communicate dataset attributes.
        varr = zarray.fill_value

    return Variable(data=varr, dims=dims, attrs=zattrs)


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
    sanitized = d.copy()
    for k, v in d.items():
        if k.startswith("."):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = ujson.loads(v)

    return cast(KerchunkArrRefs, sanitized)
