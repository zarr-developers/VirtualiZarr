import importlib.util
from pathlib import Path
from typing import List, NewType, Optional, Tuple, Union, cast

import ujson  # type: ignore
import xarray as xr

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

    if filetype.lower() == "netcdf3":
        from kerchunk.netCDF3 import NetCDF3ToZarr
        refs = NetCDF3ToZarr(filepath).translate()

    elif filetype.lower() == "netcdf4":
        from kerchunk.hdf import SingleHdf5ToZarr

        refs = SingleHdf5ToZarr(filepath).translate()
    elif filetype == "grib":
        # TODO Grib files should be handled as a DataTree object
        # see https://github.com/TomNicholas/VirtualiZarr/issues/11
        raise NotImplementedError(f"Unsupported file type: {filetype}")
    elif filetype.lower() == "tiff":
        from kerchunk.tiff import tiff_to_zarr

        refs = tiff_to_zarr(filepath)
    elif filetype.lower() == "fits":
        from kerchunk.fits import process_file

        refs = process_file(filepath)
    else:
        raise NotImplementedError(f"Unsupported file type: {filetype}")

    # TODO validate the references that were read before returning?
    return refs


def _automatically_determine_filetype(filepath: str) -> str:
    file_extension = Path(filepath).suffix

    if file_extension == ".nc":
        # checks if netCDF library is installed.
        # It currently is not a requirement in the pyproj.toml.

        if importlib.util.find_spec("netCDF4") is None:
            raise ImportError(
                "netCDF4 library is required to determine NetCDF file type."
            )

        import netCDF4

        with netCDF4.Dataset(filepath, "r") as dataset:
            if dataset.data_model == "NETCDF4":
                filetype = "netCDF4"
            elif dataset.data_model == "NETCDF3_CLASSIC":
                filetype = "netCDF3"
            else:
                raise NotImplementedError(
                    ".nc file does not appear to be NETCDF3 OR NETCDF4"
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
) -> Tuple[dict, ZArray, ZAttrs]:
    zarray = ZArray.from_kerchunk_refs(arr_refs.pop(".zarray"))
    zattrs = arr_refs.pop(".zattrs")
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
        arr_refs = variable_to_kerchunk_arr_refs(var)

        prepended_with_var_name = {
            f"{var_name}/{key}": val for key, val in arr_refs.items()
        }

        all_arr_refs.update(prepended_with_var_name)

    ds_refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            **all_arr_refs,
        },
    }

    return cast(KerchunkStoreRefs, ds_refs)


def variable_to_kerchunk_arr_refs(var: xr.Variable) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps a ManifestArray).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).
    """
    from virtualizarr.manifests import ManifestArray

    marr = var.data

    if not isinstance(marr, ManifestArray):
        raise TypeError(
            f"Can only serialize wrapped arrays of type ManifestArray, but got type {type(marr)}"
        )

    arr_refs: dict[str, Union[str, List[Union[str, int]]]] = {
        str(chunk_key): chunk_entry.to_kerchunk()
        for chunk_key, chunk_entry in marr.manifest.entries.items()
    }

    zarray_dict = marr.zarray.to_kerchunk_json()
    arr_refs[".zarray"] = zarray_dict

    zattrs = var.attrs
    zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
    arr_refs[".zattrs"] = ujson.dumps(zattrs)

    return cast(KerchunkArrRefs, arr_refs)
