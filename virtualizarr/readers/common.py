import warnings
from abc import ABC
from collections.abc import Iterable, Mapping, MutableMapping
from typing import (
    Any,
    Hashable,
    Optional,
)

import xarray  # noqa
from xarray import (
    Coordinates,
    Dataset,
    DataTree,
    Index,
    IndexVariable,
    Variable,
    open_dataset,
)
from xarray.core.indexes import PandasIndex

from virtualizarr.utils import _FsspecFSFromFilepath


def open_loadable_vars_and_indexes(
    filepath: str,
    loadable_variables,
    reader_options,
    drop_variables,
    indexes,
    group,
    decode_times,
) -> tuple[Mapping[str, Variable], Mapping[str, Index]]:
    """
    Open selected variables and indexes using xarray.

    Relies on xr.open_dataset and its auto-detection of filetypes to find the correct installed backend.
    """

    # TODO get rid of this if?
    if indexes is None or len(loadable_variables) > 0:
        # TODO we are reading a bunch of stuff we know we won't need here, e.g. all of the data variables...
        # TODO it would also be nice if we could somehow consolidate this with the reading of the kerchunk references
        # TODO really we probably want a dedicated xarray backend that iterates over all variables only once
        fpath = _FsspecFSFromFilepath(
            filepath=filepath, reader_options=reader_options
        ).open_file()

        # fpath can be `Any` thanks to fsspec.filesystem(...).open() returning Any.
        ds = open_dataset(
            fpath,  # type: ignore[arg-type]
            drop_variables=drop_variables,
            group=group,
            decode_times=decode_times,
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

        # TODO we should drop these earlier by using drop_variables
        loadable_vars = {
            str(name): var
            for name, var in ds.variables.items()
            if name in loadable_variables
        }

        # if we only read the indexes we can just close the file right away as nothing is lazy
        if loadable_vars == {}:
            ds.close()
    else:
        loadable_vars = {}
        indexes = {}

    return loadable_vars, indexes


def construct_virtual_dataset(
    virtual_vars,
    loadable_vars,
    indexes,
    coord_names,
    attrs,
) -> Dataset:
    """Construct a virtual Datset from consistuent parts."""

    vars = {**virtual_vars, **loadable_vars}

    data_vars, coords = separate_coords(vars, indexes, coord_names)

    vds = Dataset(
        data_vars,
        coords=coords,
        # indexes={},  # TODO should be added in a later version of xarray
        attrs=attrs,
    )

    # TODO we should probably also use vds.set_close() to tell xarray how to close the file we opened

    return vds


def separate_coords(
    vars: Mapping[str, Variable],
    indexes: MutableMapping[str, Index],
    coord_names: Iterable[str] | None = None,
) -> tuple[dict[str, Variable], Coordinates]:
    """
    Try to generate a set of coordinates that won't cause xarray to automatically build a pandas.Index for the 1D coordinates.

    Currently requires this function as a workaround unless xarray PR #8124 is merged.

    Will also preserve any loaded variables and indexes it is passed.
    """

    if coord_names is None:
        coord_names = []

    # split data and coordinate variables (promote dimension coordinates)
    data_vars = {}
    coord_vars: dict[
        str, tuple[Hashable, Any, dict[Any, Any], dict[Any, Any]] | Variable
    ] = {}
    found_coord_names: set[str] = set()
    # Search through variable attributes for coordinate names
    for var in vars.values():
        if "coordinates" in var.attrs:
            found_coord_names.update(var.attrs["coordinates"].split(" "))
    for name, var in vars.items():
        if name in coord_names or var.dims == (name,) or name in found_coord_names:
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

    coords = Coordinates(coord_vars, indexes=indexes)

    return data_vars, coords


class VirtualBackend(ABC):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        raise NotImplementedError()

    @staticmethod
    def open_virtual_datatree(
        path: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> DataTree:
        raise NotImplementedError()
