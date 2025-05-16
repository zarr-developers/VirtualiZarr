from __future__ import annotations

from collections.abc import Iterable, Mapping

import xarray as xr
import xarray.indexes
from obstore.store import ObjectStore

from virtualizarr.v2.protocols import Parser


def open_virtual_dataset(
    filepath: str,
    object_reader: ObjectStore,
    parser: Parser,
    drop_variables: Iterable[str] | None = None,
    loadable_variables: Iterable[str] | None = None,
    decode_times: bool | None = None,
    indexes: Mapping[str, xr.Index] | None = None,
) -> xr.Dataset:
    _drop_vars: Iterable[str] = [] if drop_variables is None else list(drop_variables)

    manifest_store = parser(
        filepath=filepath,
        object_reader=object_reader,
    )

    vds = manifest_store.to_virtual_dataset(
        loadable_variables=loadable_variables,
        decode_times=decode_times,
        indexes=indexes,
    )
    return vds.drop_vars(_drop_vars)
