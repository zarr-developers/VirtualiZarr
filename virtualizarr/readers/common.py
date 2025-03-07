from abc import ABC
from collections.abc import Iterable, Mapping
from typing import (
    Hashable,
    Any,
    MutableMapping,
    Optional,
)

import xarray as xr

from virtualizarr.utils import _FsspecFSFromFilepath


def construct_fully_virtual_dataset(
    virtual_vars: Mapping[str, xr.Variable],
    coord_names: Iterable[str] | None = None,
    attrs: dict[str, Any] = None,
) -> xr.Dataset:
    """Construct a fully virtual Dataset from constituent parts."""

    data_vars, coords = separate_coords(
        vars=virtual_vars,
        indexes={},  # we specifically avoid creating any indexes yet to avoid loading any data
        coord_names=coord_names,
    )

    vds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
    )

    # TODO we should probably also use vds.set_close() to tell xarray how to close the file we opened

    return vds


# TODO reimplement this using ManifestStore (GH #473)
def replace_virtual_with_loadable_vars(
    fully_virtual_dataset: xr.Dataset,
    filepath: str,  # TODO won't need this after #473
    group: str | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    indexes: Mapping[str, xr.Index] | None = None,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
    if indexes is not None:
        raise NotImplementedError()

    fpath = _FsspecFSFromFilepath(
        filepath=filepath, reader_options=reader_options
    ).open_file()

    # TODO replace with only opening specific variables via `open_zarr(ManifestStore)` in #473
    loadable_ds = xr.open_dataset(
        fpath,
        group=group,
        decode_times=decode_times,
    )

    if isinstance(loadable_variables, list):
        # this will automatically keep any IndexVariables needed for loadable 1D coordinates
        ds_loadable_to_keep = loadable_ds[loadable_variables]
        ds_virtual_to_keep = fully_virtual_dataset.drop_vars(loadable_variables, errors='ignore')
    elif loadable_variables is None:
        # TODO if loadable_variables is None then we have to explicitly match default behaviour of xarray
        # i.e. load and create indexes only for dimension coordinate variables
        raise NotImplementedError()
    else:
        raise ValueError()

    return xr.merge(
        [
            ds_loadable_to_keep,
            ds_virtual_to_keep,
        ]
    )


# TODO this probably doesn't need to actually support indexes != {}
def separate_coords(
    vars: Mapping[str, xr.Variable],
    indexes: MutableMapping[str, xr.Index],
    coord_names: Iterable[str] | None = None,
) -> tuple[dict[str, xr.Variable], xr.Coordinates]:
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
        str, tuple[Hashable, Any, dict[Any, Any], dict[Any, Any]] | xr.Variable
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

                if isinstance(var, xr.IndexVariable):
                    # unless variable actually already is a loaded IndexVariable,
                    # in which case we need to keep it and add the corresponding indexes explicitly
                    coord_vars[str(name)] = var
                    # TODO this seems suspect - will it handle datetimes?
                    indexes[name] = xr.PandasIndex(var, dim1d)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

    coords = xr.Coordinates(coord_vars, indexes=indexes)

    return data_vars, coords


# TODO move this into a separate api.py module
class VirtualBackend(ABC):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, xr.Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> xr.Dataset:
        raise NotImplementedError()

    @staticmethod
    def open_virtual_datatree(
        path: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, xr.Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> xr.DataTree:
        raise NotImplementedError()
