import os
import warnings
from collections.abc import Iterable, Mapping
from enum import Enum, auto
from functools import partial
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

from xarray import DataArray, Dataset, Index, combine_by_coords
from xarray.backends.api import _multi_file_closer
from xarray.backends.common import _find_absolute_paths
from xarray.core.combine import _infer_concat_order_from_positions, _nested_combine

from virtualizarr.manifests import ManifestArray
from virtualizarr.readers import (
    DMRPPVirtualBackend,
    FITSVirtualBackend,
    HDF5VirtualBackend,
    KerchunkVirtualBackend,
    NetCDF3VirtualBackend,
    TIFFVirtualBackend,
    ZarrV3VirtualBackend,
)
from virtualizarr.readers.common import VirtualBackend
from virtualizarr.utils import _FsspecFSFromFilepath, check_for_collisions

if TYPE_CHECKING:
    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
        NestedSequence,
    )


# TODO add entrypoint to allow external libraries to add to this mapping
VIRTUAL_BACKENDS = {
    "kerchunk": KerchunkVirtualBackend,
    "zarr_v3": ZarrV3VirtualBackend,
    "dmrpp": DMRPPVirtualBackend,
    # all the below call one of the kerchunk backends internally (https://fsspec.github.io/kerchunk/reference.html#file-format-backends)
    "hdf5": HDF5VirtualBackend,
    "netcdf4": HDF5VirtualBackend,  # note this is the same as for hdf5
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
    filetype: FileType | str | None = None,
    group: str | None = None,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    cftime_variables: Iterable[str] | None = None,
    indexes: Mapping[str, Index] | None = None,
    virtual_array_class=ManifestArray,
    virtual_backend_kwargs: Optional[dict] = None,
    reader_options: Optional[dict] = None,
    backend: Optional[VirtualBackend] = None,
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
        Can be one of {'netCDF3', 'netCDF4', 'HDF', 'TIFF', 'GRIB', 'FITS', 'dmrpp', 'zarr_v3', 'kerchunk'}.
        If not provided will attempt to automatically infer the correct filetype from header bytes.
    group : str, default is None
        Path to the HDF5/netCDF4 group in the given file to open. Given as a str, supported by filetypes “netcdf4”, “hdf5”, and "dmrpp".
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
    virtual_backend_kwargs: dict, default is None
        Dictionary of keyword arguments passed down to this reader. Allows passing arguments specific to certain readers.
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

    drop_variables, loadable_variables = check_for_collisions(
        drop_variables,
        loadable_variables,
    )

    if virtual_array_class is not ManifestArray:
        raise NotImplementedError()

    if reader_options is None:
        reader_options = {}

    if backend and filetype:
        raise ValueError("Cannot pass both a filetype and an explicit VirtualBackend")

    if filetype is not None:
        # if filetype is user defined, convert to FileType
        filetype = FileType(filetype)
    else:
        filetype = automatically_determine_filetype(
            filepath=filepath, reader_options=reader_options
        )
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
    paths: str | Sequence[str | os.PathLike] | NestedSequence[str | os.PathLike],
    concat_dim: (
        str
        | DataArray
        | Index
        | Sequence[str]
        | Sequence[DataArray]
        | Sequence[Index]
        | None
    ) = None,
    compat: CompatOptions = "no_conflicts",
    preprocess: Callable[[Dataset], Dataset] | None = None,
    data_vars: Literal["all", "minimal", "different"] | list[str] = "all",
    coords="different",
    combine: Literal["by_coords", "nested"] = "by_coords",
    parallel: Literal["lithops", "dask", False] = False,
    join: JoinOptions = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: CombineAttrsOptions = "override",
    **kwargs,
) -> Dataset:
    """Open multiple files as a single virtual dataset

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
    parallel : 'dask', 'lithops', or False
        Specify whether the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``, in parallel using ``lithops.map``, or in serial.
        Default is False.
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

    # TODO add options passed to open_virtual_dataset explicitly?

    paths = _find_absolute_paths(paths)

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

    if parallel == "dask":
        import dask

        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(open_virtual_dataset)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    elif parallel == "lithops":
        import lithops

        # TODO use RetryingFunctionExecutor instead?
        # TODO what's the easiest way to pass the lithops config in?
        fn_exec = lithops.FunctionExecutor()

        # lithops doesn't have a delayed primitive
        open_ = open_virtual_dataset
        # TODO I don't know how best to chain this with the getattr, or if that closing stuff is even necessary for virtual datasets
        # getattr_ = getattr
    elif parallel is not False:
        raise ValueError(
            f"{parallel} is an invalid option for the keyword argument ``parallel``"
        )
    else:
        open_ = open_virtual_dataset
        getattr_ = getattr

    if parallel == "dask":
        virtual_datasets = [open_(p, **kwargs) for p in paths1d]
        closers = [getattr_(ds, "_close") for ds in virtual_datasets]
        if preprocess is not None:
            virtual_datasets = [preprocess(ds) for ds in virtual_datasets]

        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        virtual_datasets, closers = dask.compute(virtual_datasets, closers)
    elif parallel == "lithops":

        def generate_refs(path):
            # allows passing the open_virtual_dataset function to lithops without evaluating it
            vds = open_(path, **kwargs)
            # TODO perhaps we should just load the loadable_vars here and close before returning?
            return vds

        futures = fn_exec.map(generate_refs, paths1d)

        # wait for all the serverless workers to finish, and send their resulting virtual datasets back to the client
        completed_futures, _ = fn_exec.wait(futures, download_results=True)
        virtual_datasets = [future.get_result() for future in completed_futures]
    elif parallel is False:
        virtual_datasets = [open_(p, **kwargs) for p in paths1d]
        closers = [getattr_(ds, "_close") for ds in virtual_datasets]
        if preprocess is not None:
            virtual_datasets = [preprocess(ds) for ds in virtual_datasets]

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
                f"{combine} is an invalid option for the keyword argument"
                " ``combine``"
            )
    except ValueError:
        for vds in virtual_datasets:
            vds.close()
        raise

    combined_vds.set_close(partial(_multi_file_closer, closers))

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, os.PathLike):
            attrs_file = cast(str, os.fspath(attrs_file))
        combined_vds.attrs = virtual_datasets[paths1d.index(attrs_file)].attrs

    # TODO should we just immediately close everything?
    # TODO We should have already read everything we're ever going to read into memory at this point

    return combined_vds
