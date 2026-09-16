from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

import numpy as np
import zarr
from xarray.backends.zarr import FillValueCoder
from zarr.abc.store import Store
from zarr.api.asynchronous import open_group as open_group_async

from virtualizarr.manifests import ManifestArray, ManifestGroup

# Builds a single array's ManifestArray from its opened zarr AsyncArray. Each parser
# closes over its own format-specific state (e.g. the store base URI, or the native
# chunk prefix and batch size) and supplies the rest via this callback.
ManifestArrayBuilder = Callable[["zarr.AsyncArray[Any]"], Awaitable[ManifestArray]]


async def construct_manifest_group_tree(
    store: Store,
    *,
    build_manifest_array: ManifestArrayBuilder,
    group: str | None = None,
    skip_variables: Iterable[str] | None = None,
) -> ManifestGroup:
    """Open a zarr group and recursively build a ManifestGroup, descending into subgroups.

    Shared by the Zarr and Icechunk parsers. The only parser-specific step — turning a
    single opened zarr array into a ManifestArray — is supplied via ``build_manifest_array``.

    Parameters
    ----------
    store
        The zarr store to read from.
    build_manifest_array
        Async callable mapping an opened zarr ``AsyncArray`` to a ManifestArray.
    group
        Path of the group to open as the root of this (sub)tree. ``None`` is the store root.
    skip_variables
        Names of arrays to exclude, applied at every level of the hierarchy.
    """
    zarr_group = await open_group_async(store=store, path=group, mode="r")
    skip = set() if skip_variables is None else set(skip_variables)

    array_keys = [key async for key in zarr_group.array_keys()]
    zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(key) for key in array_keys if key not in skip]
    )
    # zarr_group.getitem() returns AsyncArray | AsyncGroup; we filtered to array_keys
    # above, so every element is an AsyncArray in practice.
    manifest_arrays = await asyncio.gather(
        *[build_manifest_array(arr) for arr in zarr_arrays]  # type: ignore[arg-type]
    )
    arrays = {a.basename: ma for a, ma in zip(zarr_arrays, manifest_arrays)}

    # Subgroups, recursed depth-first so the full hierarchy is represented. Keep the
    # short name for the groups dict, but build the full path for the recursive open.
    group_keys = [key async for key in zarr_group.group_keys()]
    child_paths = [(key, key if not group else f"{group}/{key}") for key in group_keys]
    child_groups = await asyncio.gather(
        *[
            construct_manifest_group_tree(
                store,
                build_manifest_array=build_manifest_array,
                group=child_path,
                skip_variables=skip_variables,
            )
            for _, child_path in child_paths
        ]
    )
    groups = {name: mg for (name, _), mg in zip(child_paths, child_groups)}

    return ManifestGroup(
        arrays=arrays, groups=groups, attributes=dict(zarr_group.attrs)
    )


FillValueType = (
    int
    | float
    | bool
    | complex
    | str
    | np.integer
    | np.floating
    | np.bool_
    | np.complexfloating
    | bytes  # For fixed-length string storage
    | tuple[bytes, int]  # Structured type
)


def encode_cf_fill_value(
    fill_value: np.ndarray | np.generic,
    target_dtype: np.dtype,
) -> FillValueType:
    """
    Convert a fill value into one properly encoded for a target dtype.

    Parameters
    ----------
    fill_value
        An ndarray or value.
    target_dtype
        The target dtype of the ManifestArray that will use `fill_value` as its fill value.
    """
    if isinstance(fill_value, (np.ndarray, np.generic)):
        if isinstance(fill_value, np.ndarray) and fill_value.size > 1:
            raise ValueError("Expected a scalar")
        fillvalue = fill_value.item()
    else:
        fillvalue = fill_value
    encoded_fillvalue = FillValueCoder.encode(fillvalue, target_dtype)
    return encoded_fillvalue
