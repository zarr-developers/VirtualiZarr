import os
import warnings
from collections.abc import Iterable, Mapping, MutableMapping
from enum import Enum, auto
from io import BufferedIOBase
from typing import (
    Any,
    Hashable,
    Optional,
    cast,
)

import xarray as xr
from xarray.backends import AbstractDataStore, BackendArray
from xarray.coding.times import CFDatetimeCoder
from xarray.core.indexes import Index, PandasIndex
from xarray.core.variable import IndexVariable

from virtualizarr.manifests import ManifestArray
from virtualizarr.utils import _fsspec_openfile_from_filepath

XArrayOpenT = str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore


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


class ManifestBackendArray(ManifestArray, BackendArray):
    """Using this prevents xarray from wrapping the KerchunkArray in ExplicitIndexingAdapter etc."""

    ...


def open_virtual_dataset(
    filepath: str,
    *,
    filetype: FileType | None = None,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    cftime_variables: Iterable[str] | None = None,
    indexes: Mapping[str, Index] | None = None,
    virtual_array_class=ManifestArray,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
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
        Can be one of {'netCDF3', 'netCDF4', 'HDF', 'TIFF', 'GRIB', 'FITS', 'zarr_v3'}.
        If not provided will attempt to automatically infer the correct filetype from header bytes.
    drop_variables: list[str], default is None
        Variables in the file to drop before returning.
    loadable_variables: list[str], default is None
        Variables in the file to open as lazy numpy/dask arrays instead of instances of virtual_array_class.
        Default is to open all variables as virtual arrays (i.e. ManifestArray).
    cftime_variables : list[str], default is None
        Interpret the value of specified vars using cftime, returning a datetime.
        These will be automatically re-encoded with cftime. This list must be a subset
        of ``loadable_variables``.
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
    loadable_vars: dict[str, xr.Variable]
    virtual_vars: dict[str, xr.Variable]
    vars: dict[str, xr.Variable]

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

    if cftime_variables is None:
        cftime_variables = []
    elif isinstance(cftime_variables, str):
        cftime_variables = [cftime_variables]
    else:
        cftime_variables = list(cftime_variables)

    if diff := (set(cftime_variables) - set(loadable_variables)):
        missing_str = ", ".join([f"'{v}'" for v in diff])
        raise ValueError(
            "All ``cftime_variables`` must be included in ``loadable_variables`` "
            f"({missing_str} not in ``loadable_variables``)"
        )

    if virtual_array_class is not ManifestArray:
        raise NotImplementedError()

    # if filetype is user defined, convert to FileType
    if filetype is not None:
        filetype = FileType(filetype)

    if filetype == FileType.zarr_v3:
        # TODO is there a neat way of auto-detecting this?
        from virtualizarr.readers.zarr import open_virtual_dataset_from_v3_store

        return open_virtual_dataset_from_v3_store(
            storepath=filepath, drop_variables=drop_variables, indexes=indexes
        )
    elif filetype == FileType.dmrpp:
        from virtualizarr.readers.dmrpp import DMRParser

        if loadable_variables != [] or cftime_variables != [] or indexes is None:
            raise NotImplementedError(
                "Specifying `loadable_variables`, `cftime_variables` or auto-creating indexes with `indexes=None` is not supported for dmrpp files."
            )

        fpath = _fsspec_openfile_from_filepath(
            filepath=filepath, reader_options=reader_options
        )
        parser = DMRParser(fpath.read(), data_filepath=filepath.strip(".dmrpp"))
        vds = parser.parse_dataset()
        vds.drop_vars(drop_variables)
        return vds
    else:
        # we currently read every other filetype using kerchunks various file format backends
        from virtualizarr.readers.kerchunk import (
            fully_decode_arr_refs,
            read_kerchunk_references_from_file,
            virtual_vars_from_kerchunk_refs,
        )

        if reader_options is None:
            reader_options = {}

        # this is the only place we actually always need to use kerchunk directly
        # TODO avoid even reading byte ranges for variables that will be dropped later anyway?
        vds_refs = read_kerchunk_references_from_file(
            filepath=filepath,
            filetype=filetype,
            reader_options=reader_options,
        )
        virtual_vars = virtual_vars_from_kerchunk_refs(
            vds_refs,
            drop_variables=drop_variables + loadable_variables,
            virtual_array_class=virtual_array_class,
        )
        ds_attrs = fully_decode_arr_refs(vds_refs["refs"]).get(".zattrs", {})
        coord_names = ds_attrs.pop("coordinates", [])

        if indexes is None or len(loadable_variables) > 0:
            # TODO we are reading a bunch of stuff we know we won't need here, e.g. all of the data variables...
            # TODO it would also be nice if we could somehow consolidate this with the reading of the kerchunk references
            # TODO really we probably want a dedicated xarray backend that iterates over all variables only once
            fpath = _fsspec_openfile_from_filepath(
                filepath=filepath, reader_options=reader_options
            )

            # fpath can be `Any` thanks to fsspec.filesystem(...).open() returning Any.
            # We'll (hopefully safely) cast it to what xarray is expecting, but this might let errors through.

            ds = xr.open_dataset(
                cast(XArrayOpenT, fpath),
                drop_variables=drop_variables,
                decode_times=False,
            )

            if indexes is None:
                warnings.warn(
                    "Specifying `indexes=None` will create in-memory pandas indexes for each 1D coordinate, but concatenation of ManifestArrays backed by pandas indexes is not yet supported (see issue #18)."
                    "You almost certainly want to pass `indexes={}` to `open_virtual_dataset` instead."
                )

                # add default indexes by reading data from file
                indexes = {name: index for name, index in ds.xindexes.items()}
            elif indexes != {}:
                # TODO allow manual specification of index objects
                raise NotImplementedError()
            else:
                indexes = dict(**indexes)  # for type hinting: to allow mutation

            loadable_vars = {
                str(name): var
                for name, var in ds.variables.items()
                if name in loadable_variables
            }

            for name in cftime_variables:
                var = loadable_vars[name]
                loadable_vars[name] = CFDatetimeCoder().decode(var, name=name)

            # if we only read the indexes we can just close the file right away as nothing is lazy
            if loadable_vars == {}:
                ds.close()
        else:
            loadable_vars = {}
            indexes = {}

        vars = {**virtual_vars, **loadable_vars}

        decoded_vars, decoded_attrs, coord_names = determine_cf_coords(vars, ds_attrs)

        vds = construct_virtual_dataset(
            decoded_vars, indexes, decoded_attrs, coord_names
        )

        return vds


def determine_cf_coords(
    variables: T_Variables,
    attributes: T_Attrs,
) -> tuple[T_Variables, T_Attrs, set[Hashable]]:
    """
    Determines which variables are coordinate variables according to CF conventions.

    Should not actually do any decoding of values in the variables, only inspect and possibly alter their metadata.
    """
    new_vars, attrs, coord_names = decode_cf_variables(
        variables=variables,
        attributes=attributes,
        concat_characters=False,
        mask_and_scale=False,
        decode_times=False,
        decode_coords="all",
        drop_variables=None,  # should have already been dropped
        use_cftime=False,  # done separately, to only the loadable_vars
        decode_timedelta=False,  # done separately, to only the loadable_vars
    )
    return new_vars, attrs, coord_names


def construct_virtual_dataset(
    vars: Mapping[str, xr.Variable],
    indexes: MutableMapping[str, Index],
    attrs: T_Attrs,
    coord_names: Iterable[str] | None = None,
) -> xr.Dataset:
    """
    Constructs the virtual dataset but without automatically building a pandas.Index for 1D coordinates.

    Currently requires this function as a workaround unless xarray PR #8124 is merged.

    Will also preserve any loaded variables and indexes it is passed.
    """

    coord_vars: dict[str, T_VariableExpanded | xr.Variable] = {}
    data_vars = {}
    for name, var in vars.items():
        if name in coord_names:
            # use workaround to avoid creating IndexVariables described here https://github.com/pydata/xarray/pull/8107#discussion_r1311214263
            if len(var.dims) == 1:
                dim1d, *_ = var.dims
                coord_vars[name] = (dim1d, var.data, var.attrs, var.encoding)

                if isinstance(var, IndexVariable):
                    # unless variable actually already is a loaded IndexVariable,
                    # in which case we need to keep it and add the corresponding indexes explicitly
                    coord_vars[str(name)] = var
                    # TODO this seems suspect - will it handle datetimes?
                    indexes[name] = PandasIndex(var, dim1d)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

    coords = xr.Coordinates(coord_vars, indexes=indexes)

    print(indexes)

    print(coords)
    print(type(coords))

    print(data_vars)

    print(list(type(var._data) for var in data_vars.values()))
    print(list(type(var.data) for var in data_vars.values()))

    vds = xr.Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=attrs,
    )

    # TODO we should probably also use vds.set_close() to tell xarray how to close the file we opened
    # TODO see how it's done inside `xr.decode_cf`

    return vds
