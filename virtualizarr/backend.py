import warnings
from collections.abc import Iterable, Mapping
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from xarray import Dataset
from xarray.core.indexes import Index

from virtualizarr.manifests import ManifestArray
from virtualizarr.utils import _FsspecFSFromFilepath


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
    kerchunk = auto()


def automatically_determine_filetype(
    *,
    filepath: str,
    reader_options: Optional[dict[str, Any]] = {},
) -> FileType:
    """
    Attempt to automatically infer the correct reader for this filetype.

    Uses magic bytes and file / directory suffixes.
    """

    # TODO this should ideally handle every filetype that we have a reader for, not just kerchunk

    # TODO how do we handle kerchunk json / parquet here?
    if Path(filepath).suffix == ".zarr":
        # TODO we could imagine opening an existing zarr store, concatenating it, and writing a new virtual one...
        raise NotImplementedError()

    # Read magic bytes from local or remote file
    fpath = _FsspecFSFromFilepath(
        filepath=filepath, reader_options=reader_options
    ).open_file()
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


def open_virtual_dataset(
    filepath: str,
    *,
    filetype: FileType | None = None,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    cftime_variables: Iterable[str] | None = None,
    indexes: Mapping[str, Index] | None = None,
    virtual_array_class=ManifestArray,
    reader_options: Optional[dict] = None,
) -> Dataset:
    """
    Open a file or store as an xarray Dataset wrapping virtualized zarr arrays.

    No data variables will be loaded unless specified in the ``loadable_variables`` kwarg (in which case they will be xarray lazily indexed arrays).

    Xarray indexes can optionally be created (the default behaviour). To avoid creating any xarray indexes pass ``indexes={}``.

    Parameters
    ----------
    filepath : str, default None
        File path to open as a set of virtualized zarr arrays.
    filetype : FileType, default None
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        Can be one of {'netCDF3', 'netCDF4', 'HDF', 'TIFF', 'GRIB', 'FITS', 'zarr_v3', 'kerchunk'}.
        If not provided will attempt to automatically infer the correct filetype from header bytes.
    group : str, default is None
        Path to the HDF5/netCDF4 group in the given file to open. Given as a str, supported by filetypes “netcdf4” and “hdf5”.
    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    loadable_variables: list[str], default is None
        Variables in the file to open as lazy numpy/dask arrays instead of instances of virtual_array_class.
        Default is to open all variables as virtual arrays (i.e. ManifestArray).
    decode_times: bool | None, default is None
        Bool that is passed into Xarray's open_dataset. Allows time to be decoded into a datetime object.
    indexes : Mapping[str, Index], default is None
        Indexes to use on the returned xarray Dataset.
        Default is None, which will read any 1D coordinate data to create in-memory Pandas indexes.
        To avoid creating any indexes, pass indexes={}.
    virtual_array_class
        Virtual array class to use to represent the references to the chunks in each on-disk array.
        Currently can only be ManifestArray, but once VirtualZarrArray is implemented the default should be changed to that.
    reader_options: dict, default {}
        Dict passed into Kerchunk file readers, to allow reading from remote filesystems.
        Note: Each Kerchunk file reader has distinct arguments, so ensure reader_options match selected Kerchunk reader arguments.

    Returns
    -------
    vds
        An xarray Dataset containing instances of virtual_array_cls for each variable, or normal lazily indexed arrays for each variable in loadable_variables.
    """

    if cftime_variables is not None:
        # It seems like stacklevel=2 is req to surface this warning.
        warnings.warn(
            "cftime_variables is deprecated and will be ignored. Pass decode_times=True and loadable_variables=['time'] to decode time values to datetime objects.",
            DeprecationWarning,
            stacklevel=2,
        )

    if drop_variables is None:
        drop_variables = []
    elif isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    else:
        drop_variables = list(drop_variables)
    if loadable_variables is None:
        loadable_variables = []
    elif isinstance(loadable_variables, str):
        loadable_variables = [loadable_variables]
    else:
        loadable_variables = list(loadable_variables)
    common = set(drop_variables).intersection(set(loadable_variables))
    if common:
        raise ValueError(f"Cannot both load and drop variables {common}")

    if virtual_array_class is not ManifestArray:
        raise NotImplementedError()

    if reader_options is None:
        reader_options = {}

    if filetype is not None:
        # if filetype is user defined, convert to FileType
        filetype = FileType(filetype)
    else:
        filetype = automatically_determine_filetype(
            filepath=filepath, reader_options=reader_options
        )

    # TODO define these through a mapping to registered pluggable entrypoints instead
    match filetype.name.lower():
        case "kerchunk":
            from virtualizarr.readers.kerchunk import KerchunkVirtualBackend

            return KerchunkVirtualBackend.open_virtual_dataset(filepath, reader_options)

        case "zarr_v3":
            from virtualizarr.readers.zarr_v3 import ZarrV3VirtualBackend

            return ZarrV3VirtualBackend.open_virtual_dataset(
                storepath=filepath,
                drop_variables=drop_variables,
                indexes=indexes,
            )

        case "dmrpp":
            from virtualizarr.readers.dmrpp import DMRPPVirtualBackend

            return DMRPPVirtualBackend.open_virtual_dataset(
                filepath,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                indexes=indexes,
                reader_options=reader_options,
            )

        case "netcdf3":
            from virtualizarr.readers.netcdf3 import NetCDF3VirtualBackend

            return NetCDF3VirtualBackend.open_virtual_dataset(
                filepath,
                group=group,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                indexes=indexes,
                reader_options=reader_options,
            )

        case "hdf5" | "netcdf4":
            from virtualizarr.readers.hdf5 import HDF5VirtualBackend

            return HDF5VirtualBackend.open_virtual_dataset(
                filepath,
                group=group,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                indexes=indexes,
                reader_options=reader_options,
            )

        case "grib":
            # TODO Grib files should be handled as a DataTree object
            # see https://github.com/TomNicholas/VirtualiZarr/issues/11
            raise NotImplementedError(f"Unsupported file type: {filetype}")

        case "tiff":
            from virtualizarr.readers.tiff import TIFFVirtualBackend

            return TIFFVirtualBackend.open_virtual_dataset(
                filepath,
                group=group,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                indexes=indexes,
                reader_options=reader_options,
            )

        case "fits":
            from virtualizarr.readers.fits import FITSVirtualBackend

            return FITSVirtualBackend.open_virtual_dataset(
                filepath,
                group=group,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                indexes=indexes,
                reader_options=reader_options,
            )
        case _:
            raise NotImplementedError(f"Unsupported file type: {filetype.name}")

