from collections.abc import Iterable, Mapping
from typing import (
    Any,
    Hashable,
    MutableMapping,
    Optional,
)

import xarray as xr
import xarray.indexes

from virtualizarr.utils import _FsspecFSFromFilepath


def construct_fully_virtual_dataset(
    virtual_vars: Mapping[str, xr.Variable],
    coord_names: Iterable[str] | None = None,
    attrs: dict[str, Any] | None = None,
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
    filepath: str,  # TODO won't need this after #473 because the filepaths will be in the ManifestStore
    group: str | None = None,
    loadable_variables: Iterable[Hashable] | None = None,
    decode_times: bool | None = None,
    indexes: Mapping[str, xr.Index] | None = None,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
    if indexes is not None:
        raise NotImplementedError()

    fpath = _FsspecFSFromFilepath(filepath=filepath, reader_options=reader_options)

    if fpath.upath.suffix == ".zarr":
        loadable_ds = xr.open_zarr(
            fpath.upath,
            consolidated=False,
            group=group,
            decode_times=decode_times,
            chunks=None,
        )
    else:
        # TODO replace with only opening specific variables via `open_zarr(ManifestStore)` in #473
        loadable_ds = xr.open_dataset(
            fpath.open_file(),  # type: ignore[arg-type]
            group=group,
            decode_times=decode_times,
        )

    var_names_to_load: list[Hashable]
    if isinstance(loadable_variables, list):
        var_names_to_load = list(loadable_variables)
    elif loadable_variables is None:
        # If `loadable_variables`` is None then we have to explicitly match default behaviour of xarray
        # i.e. load and create indexes only for dimension coordinate variables.
        # We already have all the indexes and variables we should be keeping - we just need to distinguish them.
        var_names_to_load = [
            name for name, var in loadable_ds.variables.items() if var.dims == (name,)
        ]
    else:
        raise ValueError(
            f"loadable_variables must be an iterable of string variable names, or None, but got type {type(loadable_variables)}"
        )

    # this will automatically keep any IndexVariables needed for loadable 1D coordinates
    loadable_var_names_to_drop = set(loadable_ds.variables).difference(
        var_names_to_load
    )
    ds_loadable_to_keep = loadable_ds.drop_vars(
        loadable_var_names_to_drop, errors="ignore"
    )

    ds_virtual_to_keep = fully_virtual_dataset.drop_vars(
        var_names_to_load, errors="ignore"
    )

    # we don't need `compat` or `join` kwargs here because there should be no variables with the same name in both datasets
    return xr.merge(
        [
            ds_loadable_to_keep,
            ds_virtual_to_keep,
        ],
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
                    indexes[name] = xarray.indexes.PandasIndex(var, dim1d)
            else:
                coord_vars[name] = var
        else:
            data_vars[name] = var

    coords = xr.Coordinates(coord_vars, indexes=indexes)

    return data_vars, coords
