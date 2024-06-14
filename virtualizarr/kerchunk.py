import base64
import json
from enum import Enum, auto
from pathlib import Path
from typing import Any, NewType, Optional, cast

import numpy as np
import ujson  # type: ignore
import xarray as xr

from virtualizarr.manifests.manifest import join
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
    netcdf4 = auto()
    grib = auto()
    tiff = auto()
    fits = auto()
    zarr = auto()


class NumpyEncoder(json.JSONEncoder):
    # TODO I don't understand how kerchunk gets around this problem of encoding numpy types (in the zattrs) whilst only using ujson
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        return json.JSONEncoder.default(self, obj)


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
    reader_options: dict, default {'storage_options':{'key':'', 'secret':'', 'anon':True}}
        Dict passed into Kerchunk file readers. Note: Each Kerchunk file reader has distinct arguments,
        so ensure reader_options match selected Kerchunk reader arguments.
    """

    if filetype is None:
        filetype = _automatically_determine_filetype(
            filepath=filepath, reader_options=reader_options
        )

    if reader_options is None:
        reader_options = {}

    # if filetype is user defined, convert to FileType
    filetype = FileType(filetype)

    if filetype.name.lower() == "netcdf3":
        from kerchunk.netCDF3 import NetCDF3ToZarr

        refs = NetCDF3ToZarr(filepath, inline_threshold=0, **reader_options).translate()

    elif filetype.name.lower() == "netcdf4":
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

        refs = tiff_to_zarr(filepath, inline_threshold=0, **reader_options)
    elif filetype.name.lower() == "fits":
        from kerchunk.fits import process_file

        refs = process_file(filepath, inline_threshold=0, **reader_options)
    else:
        raise NotImplementedError(f"Unsupported file type: {filetype.name}")

    # TODO validate the references that were read before returning?
    return refs


def _automatically_determine_filetype(
    *,
    filepath: str,
    reader_options: Optional[dict[str, Any]] = None,
) -> FileType:
    file_extension = Path(filepath).suffix
    fpath = _fsspec_openfile_from_filepath(
        filepath=filepath, reader_options=reader_options
    )

    if file_extension in [".nc",".nc4",".hdf",".h5"]:
        # based off of: https://github.com/TomNicholas/VirtualiZarr/pull/43#discussion_r1543415167
        magic = fpath.read()

        if magic[0:3] == b"CDF":
            filetype = FileType.netcdf3
        elif magic[1:4] == b"HDF":
            filetype = FileType.netcdf4
        else:
            raise ValueError(".nc file does not appear to be NETCDF3 OR NETCDF4")
    elif file_extension == ".zarr":
        # TODO we could imagine opening an existing zarr store, concatenating it, and writing a new virtual one...
        raise NotImplementedError()
    elif file_extension == ".grib":
        filetype = FileType.grib
    elif file_extension in [".tif",".tiff"]:
        filetype = FileType.tiff
    elif file_extension == ".fits":
        filetype = FileType.fits
    else:
        raise NotImplementedError(f"Unrecognised file extension: {file_extension}")

    fpath.close()
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


def dataset_to_kerchunk_refs(ds: xr.Dataset) -> KerchunkStoreRefs:
    """
    Create a dictionary containing kerchunk-style store references from a single xarray.Dataset (which wraps ManifestArray objects).
    """

    all_arr_refs = {}
    for var_name, var in ds.variables.items():
        arr_refs = variable_to_kerchunk_arr_refs(var, var_name)

        prepended_with_var_name = {
            f"{var_name}/{key}": val for key, val in arr_refs.items()
        }

        all_arr_refs.update(prepended_with_var_name)

    ds_refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": ujson.dumps(ds.attrs),
            **all_arr_refs,
        },
    }

    return cast(KerchunkStoreRefs, ds_refs)


def variable_to_kerchunk_arr_refs(var: xr.Variable, var_name: str) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps either a ManifestArray or a numpy array).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).
    """
    from virtualizarr.manifests import ManifestArray

    if isinstance(var.data, ManifestArray):
        marr = var.data

        arr_refs: dict[str, str | list[str | int]] = {
            str(chunk_key): chunk_entry.to_kerchunk()
            for chunk_key, chunk_entry in marr.manifest.entries.items()
        }

        zarray = marr.zarray

    else:
        try:
            np_arr = var.to_numpy()
        except AttributeError as e:
            raise TypeError(
                f"Can only serialize wrapped arrays of type ManifestArray or numpy.ndarray, but got type {type(var.data)}"
            ) from e

        if var.encoding:
            if "scale_factor" in var.encoding:
                raise NotImplementedError(
                    f"Cannot serialize loaded variable {var_name}, as it is encoded with a scale_factor"
                )
            if "offset" in var.encoding:
                raise NotImplementedError(
                    f"Cannot serialize loaded variable {var_name}, as it is encoded with an offset"
                )
            if "calendar" in var.encoding:
                raise NotImplementedError(
                    f"Cannot serialize loaded variable {var_name}, as it is encoded with a calendar"
                )

        # This encoding is what kerchunk does when it "inlines" data, see https://github.com/fsspec/kerchunk/blob/a0c4f3b828d37f6d07995925b324595af68c4a19/kerchunk/hdf.py#L472
        byte_data = np_arr.tobytes()
        # TODO do I really need to encode then decode like this?
        inlined_data = (b"base64:" + base64.b64encode(byte_data)).decode("utf-8")

        # TODO can this be generalized to save individual chunks of a dask array?
        # TODO will this fail for a scalar?
        arr_refs = {join(0 for _ in np_arr.shape): inlined_data}

        zarray = ZArray(
            chunks=np_arr.shape,
            shape=np_arr.shape,
            dtype=np_arr.dtype,
            order="C",
        )

    zarray_dict = zarray.to_kerchunk_json()
    arr_refs[".zarray"] = zarray_dict

    zattrs = var.attrs
    zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
    arr_refs[".zattrs"] = json.dumps(zattrs, separators=(",", ":"), cls=NumpyEncoder)

    return cast(KerchunkArrRefs, arr_refs)
