from abc import ABC
from collections.abc import Iterable, Mapping
from typing import (
    Optional,
)

import xarray as xr

from virtualizarr.utils import _FsspecFSFromFilepath


def construct_fully_virtual_dataset(
    virtual_vars,
    coord_names,
    attrs,
) -> xr.Dataset:
    """Construct a fully virtual Dataset from constituent parts."""

    data_vars = {name: var for name, var in virtual_vars.items() if name not in coord_names}
    coord_vars = {name: var for name, var in virtual_vars.items() if name in coord_names}
    
    # We avoid constructing indexes yet so as to delay loading any data.
    coords = xr.Coordinates(coords=coord_vars, indexes={})

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
        ds_loadable_to_keep = loadable_ds[loadable_variables]
        ds_virtual_to_keep = fully_virtual_dataset.drop(loadable_variables)
    elif loadable_variables is None:
        # TODO if loadable_variables is None then we have to explicitly match default behaviour of xarray
        raise NotImplementedError()
    else:
        raise ValueError()

    return xr.merge(
        [
            ds_loadable_to_keep, 
            ds_virtual_to_keep,
        ]
    )


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
