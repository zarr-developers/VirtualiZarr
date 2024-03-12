import json
from pathlib import Path
from typing import Literal, NewType, Optional, Tuple

import kerchunk
import xarray as xr
from pydantic import BaseModel, ConfigDict

from virtualizarr.zarr import ZArray, ZAttrs

# Distinguishing these via type hints makes it a lot easier to mentally keep track of what the opaque kerchunk "reference dicts" actually mean
# (idea from https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html)
KerchunkStoreRefs = NewType(
    "KerchunkStoreRefs", dict[Literal["version"] | Literal["refs"], int | dict]
)  # top-level dict with keys for 'version', 'refs'
KerchunkArrRefs = NewType(
    "KerchunkArrRefs",
    dict[Literal[".zattrs"] | Literal[".zarray"] | str, ZAttrs | ZArray | str],
)  # lower-level dict containing just the information for one zarr array


class KerchunkChunkEntry(BaseModel):
    """Like a ChunkEntry but follows kerchunks' specification"""

    model_config = ConfigDict(frozen=True)

    entry: Tuple[str, int, int]


class KerchunkChunkDict(BaseModel):
    """Like a ChunkManifest but follows kerchunks' specification"""

    model_config = ConfigDict(frozen=True)


def read_kerchunk_references_from_file(
    filepath: str, filetype: Optional[str]
) -> KerchunkStoreRefs:
    """
    Read a single legacy file and return kerchunk references to its contents.

    Parameters
    ----------
    filepath : str, default: None
        File path to open as a set of virtualized zarr arrays.
    filetype : str, default: None
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        If not provided will attempt to automatically infer the correct filetype from the the filepath's extension.
    """

    if filetype is None:
        filetype = _automatically_determine_filetype(filepath)

    if filetype == "netCDF3":
        refs = kerchunk.netCDF3.NetCDF3ToZarr(filepath).translate()
    elif filetype == "netCDF4":
        refs = kerchunk.hdf.SingleHdf5ToZarr(filepath).translate()
    elif filetype == "grib":
        # TODO Grib files should be handled as a DataTree object
        # see https://github.com/TomNicholas/VirtualiZarr/issues/11
        raise NotImplementedError(f"Unsupported file type: {filetype}")
    elif filetype == "tiff":
        refs = kerchunk.tiff.tiff_to_zarr(filepath)
    elif filetype == "fits":
        refs = kerchunk.fits.process_file(filepath)
    else:
        raise NotImplementedError(f"Unsupported file type: {filetype}")

    # TODO validate the references that were read before returning?
    return refs


def _automatically_determine_filetype(filepath: str) -> str:
    file_extension = Path(filepath).suffix

    if file_extension == ".nc":
        # TODO how can we automatically distinguish netCDF3 and 4?
        raise NotImplementedError(
            "Cannot unambiguously automatically determine which kerchunk file format reader to use"
        )
    elif file_extension == ".zarr":
        # TODO we could imagine opening an existing zarr store, concatenating it, and writing a new virtual one...
        raise NotImplementedError()
    elif file_extension == ".grib":
        filetype = "grib"
    elif file_extension == ".tiff":
        filetype = "tiff"
    elif file_extension == ".fits":
        filetype = "fits"
    else:
        raise NotImplementedError(f"Unrecognised file extension: {file_extension}")

    return filetype


def find_var_names(ds_reference_dict: KerchunkStoreRefs) -> list[str]:
    """Find the names of zarr variables in this store/group."""

    refs = ds_reference_dict["refs"]
    found_var_names = [key.split("/")[0] for key in refs.keys() if "/" in key]
    return found_var_names


def extract_array_refs(
    ds_reference_dict: KerchunkStoreRefs, var_name: str
) -> KerchunkArrRefs:
    """Extract only the part of the kerchunk reference dict that is relevant to this one zarr array"""

    found_var_names = find_var_names(ds_reference_dict)

    refs = ds_reference_dict["refs"]
    if var_name in found_var_names:
        arr_refs = {
            key.split("/")[1]: refs[key]
            for key in refs.keys()
            if var_name == key.split("/")[0]
        }

        # TODO return this separately?
        # zattrs = var_refs.pop(".zattrs")  # we are going to store these separately later

        return fully_decode_arr_refs(arr_refs)
    else:
        raise KeyError(
            f"Could not find zarr array variable name {var_name}, only {found_var_names}"
        )


def parse_array_refs(
    arr_refs: KerchunkArrRefs,
) -> Tuple[KerchunkChunkDict, ZArray, ZAttrs]:
    zarray = ZArray.from_kerchunk_refs(arr_refs.pop(".zarray"))
    zattrs = arr_refs.pop(".zattrs")
    chunk_dict = arr_refs

    return chunk_dict, zarray, zattrs


def fully_decode_arr_refs(d: KerchunkArrRefs) -> KerchunkArrRefs:
    """
    Only have to do this because kerchunk.SingleHdf5ToZarr apparently doesn't bother converting .zarray and .zattrs contents to dicts, see https://github.com/fsspec/kerchunk/issues/415 .
    """
    sanitized = d.copy()
    for k, v in d.items():
        if k.startswith("."):
            # ensure contents of .zattrs and .zarray are python dictionaries
            sanitized[k] = json.loads(v)
        # TODO should we also convert the byte range values stored under chunk keys to python lists? e.g. 'time/0': ['air.nc', 7757515, 11680]

    return sanitized


def dataset_to_kerchunk_refs(ds: xr.Dataset) -> KerchunkStoreRefs:
    ...
