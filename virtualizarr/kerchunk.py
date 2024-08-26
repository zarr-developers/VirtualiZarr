import warnings
from enum import Enum, auto
from pathlib import Path
from typing import Any, NewType, Optional, cast

import ujson  # type: ignore

from virtualizarr.utils import _fsspec_openfile_from_filepath
from virtualizarr.zarr import ZArray, ZAttrs

# Distinguishing these via type hints makes it a lot easier to mentally keep track of what the opaque kerchunk "reference dicts" actually mean
# (idea from https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
# TODO I would prefer to be more specific about these types
KerchunkStoreRefs = NewType(
    "KerchunkStoreRefs", dict
)  # top-level dict with keys for 'version', 'refs'
KerchunkArrRefs = NewType(
    "KerchunkArrRefs",
    dict,
)  # lower-level dict containing just the information for one zarr array


class AutoName(Enum):
    # Recommended by official Python docs for auto naming:
    # https://docs.python.org/3/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class FileType(AutoName):
    netcdf3 = auto()
    netcdf4 = auto()  # NOTE: netCDF4 is a subset of hdf5
    hdf4 = auto()
    hdf5 = auto()
    grib = auto()
    tiff = auto()
    fits = auto()
    zarr = auto()
    dmrpp = auto()
    zarr_v3 = auto()


def read_kerchunk_references_from_file(
    filepath: str,
    filetype: FileType | None,
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
    reader_options: dict, default {}
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
