from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Literal,
    Optional,
    cast,
)

import xarray as xr
import xarray.indexes
from xarray import DataArray, Dataset, Index, combine_by_coords
from xarray.backends.common import _find_absolute_paths
from xarray.core import dtypes
from xarray.core.types import NestedSequence
from xarray.structure.combine import _infer_concat_order_from_positions, _nested_combine

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.parallel import get_executor
from virtualizarr.parsers.typing import Parser
from virtualizarr.registry import ObjectStoreRegistry

if TYPE_CHECKING:
    from xarray.core.types import (
        CombineAttrsOptions,
        CompatOptions,
        JoinOptions,
    )


def open_virtual_dataset(
    url: str,
    registry: ObjectStoreRegistry,
    parser: Parser,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
) -> xr.Dataset:
    """
    Open an archival data source as an [xarray.Dataset][] wrapping virtualized zarr arrays.

    No data variables will be loaded unless specified in the ``loadable_variables`` kwarg (in which case they will open as lazily indexed arrays using xarray's standard lazy indexing classes).

    Xarray indexes can optionally be created (the default behaviour is to create indexes for any 1D coordinate variables). To avoid creating any xarray indexes pass ``indexes={}``.

    Parameters
    ----------
    url
        The url of the data source to virtualize. The URL should include a scheme. For example:

        - `url="file:///Users/my-name/Documents/my-project/my-data.nc"` for a local data source.
        - `url="s3://my-bucket/my-project/my-data.nc"` for a remote data source on an S3 compatible cloud.

    registry
        An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.
    parser
        A parser to use for the given data source. For example:

        - [virtualizarr.parsers.HDFParser][] for virtualizing NetCDF4 or HDF5 files.
        - [virtualizarr.parsers.FITSParser][] for virtualizing FITS files.
        - [virtualizarr.parsers.NetCDF3Parser][] for virtualizing NetCDF3 files.
        - [virtualizarr.parsers.KerchunkJSONParser][] for re-opening Kerchunk JSONs.
        - [virtualizarr.parsers.KerchunkParquetParser][] for re-opening Kerchunk Parquets.
        - [virtualizarr.parsers.ZarrParser][] for virtualizing Zarr stores.

    drop_variables
        Variables in the data source to drop before returning.
    loadable_variables
        Variables in the data source to load as Dask/NumPy arrays instead of as virtual arrays.
    decode_times
        Bool that is passed into [xarray.open_dataset][]. Allows time to be decoded into a datetime object.

    Returns
    -------
    vds
        An [xarray.Dataset][] containing virtual chunk references for all variables not included
        in `loadable_variables` and normal lazily indexed arrays for each variable in `loadable_variables`.
    """
    filepath = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())

    manifest_store = parser(
        url=filepath,
        registry=registry,
    )

    ds = manifest_store.to_virtual_dataset(
        loadable_variables=loadable_variables,
        decode_times=decode_times,
    )
    return ds.drop_vars(list(drop_variables or ()))


def open_virtual_mfdataset(
    urls: (
        str
        | os.PathLike
        | Sequence[str | os.PathLike]
        | NestedSequence[str | os.PathLike]
    ),
    registry: ObjectStoreRegistry,
    parser: Parser,
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
    parallel: Literal["dask", "lithops", False] | type[Executor] = False,
    join: "JoinOptions" = "outer",
    attrs_file: str | os.PathLike | None = None,
    combine_attrs: "CombineAttrsOptions" = "override",
    **kwargs,
) -> Dataset:
    """
    Open multiple data sources as a single virtual dataset.

    This function is explicitly modelled after [xarray.open_mfdataset][], and works in the same way.

    If `combine='by_coords'` then the function `combine_by_coords` is used to combine
    the datasets into one before returning the result, and if combine='nested' then
    `combine_nested` is used. The urls must be structured according to which
    combining function is used, the details of which are given in the documentation for
    `combine_by_coords` and `combine_nested`. By default `combine='by_coords'`
    will be used. Global attributes from the `attrs_file` are used
    for the combined dataset.

    Parameters
    ----------
    urls
        Same as in [virtualizarr.open_virtual_dataset][]
    registry
        An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.
    concat_dim
        Same as in [xarray.open_mfdataset][]
    compat
        Same as in [xarray.open_mfdataset][]
    preprocess
        Same as in [xarray.open_mfdataset][]
    data_vars
        Same as in [xarray.open_mfdataset][]
    coords
        Same as in [xarray.open_mfdataset][]
    combine
        Same as in [xarray.open_mfdataset][]
    parallel : "dask", "lithops", False, or type of subclass of [concurrent.futures.Executor][]
        Specify whether the open and preprocess steps of this function will be
        performed in parallel using [lithops][], `dask.delayed`, or any executor compatible
        with the [concurrent.futures][] interface, or in serial.
        Default is False, which will execute these steps in serial.
    join
        Same as in [xarray.open_mfdataset][]
    attrs_file
        Same as in [xarray.open_mfdataset][]
    combine_attrs
        Same as in [xarray.open_mfdataset][]
    **kwargs : optional
        Additional arguments passed on to [virtualizarr.open_virtual_dataset][]. For an
        overview of some of the possible options, see the documentation of
        [virtualizarr.open_virtual_dataset][].

    Returns
    -------
    vds
        An [xarray.Dataset][] containing virtual chunk references for all variables not included
        in `loadable_variables` and normal lazily indexed arrays for each variable in `loadable_variables`.

    Notes
    -----
    The results of opening each virtual dataset in parallel are sent back to the client process, so must not be too large. See the docs page on [Scaling][].
    """

    # TODO this is practically all just copied from xarray.open_mfdataset - an argument for writing a virtualizarr engine for xarray?

    # TODO list kwargs passed to open_virtual_dataset explicitly in docstring?

    paths = cast(NestedSequence[str], _find_absolute_paths(urls))

    if not paths:
        raise OSError("No data sources to open, pass urls to the `urls` parameter.")

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
            ds = open_virtual_dataset(
                url=path, registry=registry, parser=parser, **kwargs
            )
            return preprocess(ds)

        open_func = _open_and_preprocess
    else:

        def _open(path: str) -> xr.Dataset:
            return open_virtual_dataset(
                url=path, registry=registry, parser=parser, **kwargs
            )

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
                fill_value=dtypes.NA,
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
                f"{combine} is an invalid option for the keyword argument `combine`"
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
    manifest_store: ManifestStore,
    group: str | None = None,
    loadable_variables: Iterable[Hashable] | None = None,
    decode_times: bool | None = None,
    reader_options: Optional[dict] = None,
) -> xr.Dataset:
    """
    Construct a fully or partly virtual dataset from a ManifestStore
    containing the contents of one group.

    """

    if group:
        raise NotImplementedError("ManifestStore does not yet support nested groups")
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
