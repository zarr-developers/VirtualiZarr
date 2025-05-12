from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import (
    Any,
    Hashable,
    MutableMapping,
    Optional,
)

import xarray as xr
import xarray.indexes
from obstore.store import ObjectStore

from virtualizarr.backends import Backend
from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.utils import _FsspecFSFromFilepath


def open_virtual_dataset(
    filepath: str,
    object_reader: ObjectStore, 
    backend: Backend,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    cftime_variables: Iterable[str] | None = None,
    indexes: Mapping[str, xr.Index] | None = None,
) -> xr.Dataset:
    filepath = validate_and_normalize_path_to_uri(
        filepath, fs_root=Path.cwd().as_uri()
    )

    _drop_vars: Iterable[str] = (
        [] if drop_variables is None else list(drop_variables)
    )

    manifest_store = backend(
        filepath=filepath,
        object_reader=object_reader,
    )

    ds = manifest_store.to_virtual_dataset(
        loadable_variables=loadable_variables,
        decode_times=decode_times,
        indexes=indexes,
    )
    return ds


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

    return vds


def construct_virtual_dataset(
    manifest_store: ManifestStore | None = None,
    # TODO remove filepath option once all readers use ManifestStore approach
    fully_virtual_ds: xr.Dataset | None = None,
    filepath: str | None = None,
    group: str | None = None,
    loadable_variables: Iterable[Hashable] | None = None,
    decode_times: bool | None = None,
    indexes: Mapping[str, xr.Index] | None = None,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
    """
    Construct a fully or partly virtual dataset from a ManifestStore (or filepath for backwards compatibility),
    containing the contents of one group.

    Accepts EITHER manifest_store OR fully_virtual_ds and filepath. The latter option should be removed once all readers use ManifestStore approach.
    """

    if indexes is not None:
        raise NotImplementedError()

    if manifest_store:
        if group:
            raise NotImplementedError(
                "ManifestStore does not yet support nested groups"
            )
        else:
            manifestgroup = manifest_store._group

        fully_virtual_ds = manifestgroup.to_virtual_dataset()

        with xr.open_zarr(
            manifest_store,
            group=group,
            consolidated=False,
            zarr_format=3,
            chunks=None,
            decode_times=decode_times,
        ) as loadable_ds:
            return replace_virtual_with_loadable_vars(
                fully_virtual_ds, loadable_ds, loadable_variables
            )
    else:
        # TODO pre-ManifestStore codepath, remove once all readers use ManifestStore approach

        fpath = _FsspecFSFromFilepath(
            filepath=filepath,  # type: ignore[arg-type]
            reader_options=reader_options,
        ).open_file()

        with xr.open_dataset(
            fpath,  # type: ignore[arg-type]
            group=group,
            decode_times=decode_times,
        ) as loadable_ds:
            return replace_virtual_with_loadable_vars(
                fully_virtual_ds,  # type: ignore[arg-type]
                loadable_ds,
                loadable_variables,
            )


def replace_virtual_with_loadable_vars(
    fully_virtual_ds: xr.Dataset,
    loadable_ds: xr.Dataset,
    loadable_variables: Iterable[Hashable] | None = None,
) -> xr.Dataset:
    """
    Merge a fully virtual and the corresponding fully loadable dataset, keeping only `loadable_variables` from the latter (plus defaults needed for indexes).
    """

    var_names_to_load: list[Hashable]

    if isinstance(loadable_variables, list):
        var_names_to_load = list(loadable_variables)
    elif loadable_variables is None:
        # If `loadable_variables` is None, then we have to explicitly match default
        # behaviour of xarray, i.e., load and create indexes only for dimension
        # coordinate variables.  We already have all the indexes and variables
        # we should be keeping - we just need to distinguish them.
        var_names_to_load = [
            name for name, var in loadable_ds.variables.items() if var.dims == (name,)
        ]
    else:
        raise ValueError(
            "loadable_variables must be an iterable of string variable names,"
            f" or None, but got type {type(loadable_variables)}"
        )

    # this will automatically keep any IndexVariables needed for loadable 1D coordinates
    loadable_var_names_to_drop = set(loadable_ds.variables).difference(
        var_names_to_load
    )
    ds_loadable_to_keep = loadable_ds.drop_vars(
        loadable_var_names_to_drop, errors="ignore"
    )

    ds_virtual_to_keep = fully_virtual_ds.drop_vars(var_names_to_load, errors="ignore")

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
