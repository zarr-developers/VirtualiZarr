import os
import warnings
from collections.abc import Iterable, Mapping
from concurrent.futures import Executor
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    cast,
)

import xarray as xr
from xarray import DataArray, Dataset, Index, combine_by_coords
from xarray.backends.common import _find_absolute_paths
from xarray.core.types import NestedSequence
from xarray.structure.combine import _infer_concat_order_from_positions, _nested_combine

from virtualizarr.parallel import get_executor
from virtualizarr.readers import (
    DMRPPVirtualBackend,
    FITSVirtualBackend,
    HDFVirtualBackend,
    KerchunkVirtualBackend,
    NetCDF3VirtualBackend,
    TIFFVirtualBackend,
    ZarrVirtualBackend,
)
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.utils import _FsspecFSFromFilepath

if TYPE_CHECKING:
    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
    )


# TODO add entrypoint to allow external libraries to add to this mapping
VIRTUAL_BACKENDS = {
    "kerchunk": KerchunkVirtualBackend,
    "zarr": ZarrVirtualBackend,
    "dmrpp": DMRPPVirtualBackend,
    "hdf5": HDFVirtualBackend,
    "netcdf4": HDFVirtualBackend,  # note this is the same as for hdf5
    # all the below call one of the kerchunk backends internally (https://fsspec.github.io/kerchunk/reference.html#file-format-backends)
    "netcdf3": NetCDF3VirtualBackend,
    "tiff": TIFFVirtualBackend,
    "fits": FITSVirtualBackend,
}


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
    dmrpp = auto()
    kerchunk = auto()
    zarr = auto()


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
        return FileType.zarr

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
    filetype: FileType | str | None = None,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    cftime_variables: Iterable[str] | None = None,
    indexes: Mapping[str, Index] | None = None,
    virtual_backend_kwargs: dict | None = None,
    reader_options: dict | None = None,
    backend: type[VirtualBackend] | None = None,
) -> Dataset:
    """
    Open a file or store as an xarray.Dataset wrapping virtualized zarr arrays.

    Some variables can be opened as loadable lazy numpy arrays. This can be controlled explicitly using the ``loadable_variables`` keyword argument.
    By default this will be the same variables which `xarray.open_dataset` would create indexes for: i.e. one-dimensional coordinate variables whose
    name matches the name of their only dimension (also known as "dimension coordinates").
    Pandas indexes will also now be created by default for these loadable variables, but this can be controlled by passing a value for the ``indexes`` keyword argument.
    To avoid creating any xarray indexes pass ``indexes={}``.

    Parameters
    ----------
    filepath
        File path to open as a set of virtualized zarr arrays.
    filetype
        Type of file to be opened. Used to determine which kerchunk file format backend to use.
        Can be one of {'netCDF3', 'netCDF4', 'HDF', 'TIFF', 'GRIB', 'FITS', 'dmrpp', 'kerchunk'}.
        If not provided will attempt to automatically infer the correct filetype from header bytes.
    group
        Path to the HDF5/netCDF4 group in the given file to open. Given as a str, supported by filetypes “netcdf4”, “hdf5”, and "dmrpp".
    drop_variables
        Variables in the file to drop before returning.
    loadable_variables
        Variables in the file to open as lazy numpy/dask arrays instead of instances of `ManifestArray`.
        Default is to open all variables as virtual variables (i.e. as ManifestArrays).
    decode_times
        Bool that is passed into Xarray's open_dataset. Allows time to be decoded into a datetime object.
    indexes
        Indexes to use on the returned xarray Dataset.
        Default is None, which will read any 1D coordinate data to create in-memory Pandas indexes.
        To avoid creating any indexes, pass indexes={}.
    virtual_backend_kwargs
        Dictionary of keyword arguments passed down to this reader. Allows passing arguments specific to certain readers.
    reader_options
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

    if reader_options is None:
        reader_options = {}

    if backend and filetype:
        raise ValueError("Cannot pass both a filetype and an explicit VirtualBackend")

    if filetype is None:
        filetype = automatically_determine_filetype(
            filepath=filepath, reader_options=reader_options
        )
    elif isinstance(filetype, str):
        # if filetype is a user defined string, convert to FileType
        filetype = FileType(filetype.lower())
    elif not isinstance(filetype, FileType):
        raise ValueError("Filetype must be a valid string or FileType")

    if backend:
        backend_cls = backend
    else:
        backend_cls = VIRTUAL_BACKENDS.get(filetype.name.lower())  # type: ignore

    if backend_cls is None:
        raise NotImplementedError(f"Unsupported file type: {filetype.name}")

    vds = backend_cls.open_virtual_dataset(
        filepath,
        group=group,
        drop_variables=drop_variables,
        loadable_variables=loadable_variables,
        decode_times=decode_times,
        indexes=indexes,
        virtual_backend_kwargs=virtual_backend_kwargs,
        reader_options=reader_options,
    )

    return vds


def open_virtual_mfdataset(
    paths: str
    | os.PathLike
    | Sequence[str | os.PathLike]
    | "NestedSequence[str | os.PathLike]",
    concat_dim: (
        str
        | DataArray
        | Index
        | Sequence[str]
        | Sequence[DataArray]
        | Sequence[Index]
        | None
    ) = None,
    compat: "CompatOptions" = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: Literal["dask", "lithops", False] | Executor = False,
    join: "JoinOptions" = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: "CombineAttrsOptions" = "override",
    **kwargs,
) -> Dataset:
    """
    Open multiple files as a single virtual dataset.

    If combine='by_coords' then the function ``combine_by_coords`` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    ``combine_nested`` is used. The filepaths must be structured according to which
    combining function is used, the details of which are given in the documentation for
    ``combine_by_coords`` and ``combine_nested``. By default ``combine='by_coords'``
    will be used. Global attributes from the ``attrs_file`` are used
    for the combined dataset.

    Parameters
    ----------
    paths
        Same as in xarray.open_mfdataset
    concat_dim
        Same as in xarray.open_mfdataset
    compat
        Same as in xarray.open_mfdataset
    preprocess
        Same as in xarray.open_mfdataset
    data_vars
        Same as in xarray.open_mfdataset
    coords
        Same as in xarray.open_mfdataset
    combine
        Same as in xarray.open_mfdataset
    parallel : "dask", "lithops", False, or instance of a subclass of ``concurrent.futures.Executor``
        Specify whether the open and preprocess steps of this function will be
        performed in parallel using lithops, dask.delayed, or any executor compatible
        with the ``concurrent.futures`` interface, or in serial.
        Default is False, which will execute these steps in serial.
    join
        Same as in xarray.open_mfdataset
    attrs_file
        Same as in xarray.open_mfdataset
    combine_attrs
        Same as in xarray.open_mfdataset
    **kwargs : optional
        Additional arguments passed on to :py:func:`virtualizarr.open_virtual_dataset`. For an
        overview of some of the possible options, see the documentation of
        :py:func:`virtualizarr.open_virtual_dataset`.

    Returns
    -------
    xarray.Dataset

    Notes
    -----
    The results of opening each virtual dataset in parallel are sent back to the client process, so must not be too large.
    """

    # TODO this is practically all just copied from xarray.open_mfdataset - an argument for writing a virtualizarr engine for xarray?

    # TODO list kwargs passed to open_virtual_dataset explicitly in docstring?

    paths = cast(NestedSequence[str], _find_absolute_paths(paths))

    if not paths:
        raise OSError("no files to open")

    paths1d: list[str]
    if combine == "nested":
        if isinstance(concat_dim, str | DataArray) or concat_dim is None:
            concat_dim = [concat_dim]  # type: ignore[assignment]

        # This creates a flat list which is easier to iterate over, whilst
        # encoding the originally-supplied structure as "ids".
        # The "ids" are not used at all if combine='by_coords`.
        combined_ids_paths = _infer_concat_order_from_positions(paths)
        ids, paths1d = (
            list(combined_ids_paths.keys()),
            list(combined_ids_paths.values()),
        )
    elif concat_dim is not None:
        raise ValueError(
            "When combine='by_coords', passing a value for `concat_dim` has no "
            "effect. To manually combine along a specific dimension you should "
            "instead specify combine='nested' along with a value for `concat_dim`.",
        )
    else:
        paths1d = paths  # type: ignore[assignment]

    # TODO this refactored preprocess and executor logic should be upstreamed into xarray - see https://github.com/pydata/xarray/pull/9932

    if preprocess:
        # TODO we could reexpress these using functools.partial but then we would hit this lithops bug: https://github.com/lithops-cloud/lithops/issues/1428

        def _open_and_preprocess(path: str) -> xr.Dataset:
            ds = open_virtual_dataset(path, **kwargs)
            return preprocess(ds)

        open_func = _open_and_preprocess
    else:

        def _open(path: str) -> xr.Dataset:
            return open_virtual_dataset(path, **kwargs)

        open_func = _open

    executor = get_executor(parallel=parallel)
    with executor() as exec:
        # wait for all the workers to finish, and send their resulting virtual datasets back to the client for concatenation there
        virtual_datasets = list(
            exec.map(
                open_func,
                paths1d,
            )
        )

    # TODO add file closers

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined_vds = _nested_combine(
                virtual_datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
                combine_attrs=combine_attrs,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined_vds = combine_by_coords(
                virtual_datasets,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                combine_attrs=combine_attrs,
            )
        else:
            raise ValueError(
                f"{combine} is an invalid option for the keyword argument ``combine``"
            )
    except ValueError:
        for vds in virtual_datasets:
            vds.close()
        raise

    # combined_vds.set_close(partial(_multi_file_closer, closers))

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, os.PathLike):
            attrs_file = cast(str, os.fspath(attrs_file))
        combined_vds.attrs = virtual_datasets[paths1d.index(attrs_file)].attrs

    # TODO should we just immediately close everything?
    # TODO If loadable_variables is eager then we should have already read everything we're ever going to read into memory at this point

    return combined_vds
