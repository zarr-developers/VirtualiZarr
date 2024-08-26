
from typing import Optional, Any
from pathlib import Path
import warnings
from typing import MutableMapping

from xarray.core.variable import Variable
from xarray.core.indexes import Index
from xarray import Dataset

from virtualizarr.kerchunk import KerchunkStoreRefs, find_var_names, extract_array_refs, parse_array_refs, fully_decode_arr_refs
from virtualizarr.manifests import ManifestArray, ChunkManifest
from virtualizarr.backend import separate_coords, FileType
from virtualizarr.utils import _fsspec_openfile_from_filepath


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
