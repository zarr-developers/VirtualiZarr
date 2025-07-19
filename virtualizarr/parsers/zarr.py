from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path  # noqa
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np
from zarr.api.asynchronous import open_group as open_group_async
from zarr.core.metadata import ArrayV3Metadata
from zarr.storage import ObjectStore

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri  # noqa
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.vendor.zarr.core.common import _concurrent_map

FillValueT = bool | str | float | int | list | None

ZARR_DEFAULT_FILL_VALUE: dict[str, FillValueT] = {
    # numpy dtypes's hierarchy lets us avoid checking for all the widths
    # https://numpy.org/doc/stable/reference/arrays.scalars.html
    np.dtype("bool").kind: False,
    np.dtype("int").kind: 0,
    np.dtype("float").kind: 0.0,
    np.dtype("complex").kind: [0.0, 0.0],
    np.dtype("datetime64").kind: 0,
}

if TYPE_CHECKING:
    import zarr


async def get_chunk_mapping_prefix(zarr_array: zarr.AsyncArray, path: str) -> dict:
    """Create a dictionary to pass into ChunkManifest __init__"""

    # TODO: For when we want to support reading V2 we should parse the /c/ and "/" between chunks
    if zarr_array.shape == ():
        # If we have a scalar array `c`
        # https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/default/index.html#description

        prefix = zarr_array.name.lstrip("/") + "/c"
        prefix_keys = [(prefix,)]
        _lengths = [await zarr_array.store.getsize("c")]
        _dict_keys = ["c"]
        _paths = [path + "/" + _dict_keys[0]]

    else:
        prefix = zarr_array.name.lstrip("/") + "/c/"
        prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
        _lengths = await _concurrent_map(prefix_keys, zarr_array.store.getsize)
        chunk_keys = [x[0].split(prefix)[1] for x in prefix_keys]
        _dict_keys = [key.replace("/", ".") for key in chunk_keys]
        _paths = [path + "/" + prefix + key for key in chunk_keys]

    _offsets = [0] * len(_lengths)
    return {
        key: {"path": path, "offset": offset, "length": length}
        for key, path, offset, length in zip(
            _dict_keys,
            _paths,
            _offsets,
            _lengths,
        )
    }


async def build_chunk_manifest(zarr_array: zarr.AsyncArray, path: str) -> ChunkManifest:
    """Build a ChunkManifest from a dictionary"""
    chunk_map = await get_chunk_mapping_prefix(zarr_array=zarr_array, path=path)
    return ChunkManifest(chunk_map)


def get_metadata(zarr_array: zarr.AsyncArray[Any]) -> ArrayV3Metadata:
    fill_value = zarr_array.metadata.fill_value
    if fill_value is not None:
        fill_value = ZARR_DEFAULT_FILL_VALUE[zarr_array.metadata.fill_value.dtype.kind]

    zarr_format = zarr_array.metadata.zarr_format

    if zarr_format == 2:
        # TODO: Once we want to support V2, we will have to deconstruct the
        # zarr_array codecs etc. and reconstruct them with create_v3_array_metadata
        raise NotImplementedError("Reading Zarr V2 currently not supported.")

    elif zarr_format == 3:
        return zarr_array.metadata

    else:
        raise NotImplementedError("Zarr format is not recognized as v2 or v3.")


async def _construct_manifest_array(zarr_array: zarr.AsyncArray[Any], path: str):
    array_metadata = get_metadata(zarr_array=zarr_array)

    chunk_manifest = await build_chunk_manifest(zarr_array, path=path)
    return ManifestArray(metadata=array_metadata, chunkmanifest=chunk_manifest)


async def _construct_manifest_group(
    path: str,
    store: zarr.storage.ObjectStore,
    *,
    skip_variables: str | Iterable[str] | None = None,
    group: str | None = None,
):
    zarr_group = await open_group_async(
        store=store,
        path=group,
        mode="r",
    )

    zarr_array_keys = [key async for key in zarr_group.array_keys()]

    _skip_variables: list[Hashable] = (
        [] if skip_variables is None else list(skip_variables)
    )

    zarr_arrays = await asyncio.gather(
        *[
            zarr_group.getitem(var)
            for var in zarr_array_keys
            if var not in _skip_variables
        ]
    )
    manifest_arrays = await asyncio.gather(
        *[
            _construct_manifest_array(zarr_array=array, path=path)  # type: ignore[arg-type]
            for array in zarr_arrays
        ]
    )

    manifest_dict = {
        array.basename: result for array, result in zip(zarr_arrays, manifest_arrays)
    }

    return ManifestGroup(manifest_dict, attributes=zarr_group.attrs)


class ZarrParser:
    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the
        `__call__` method.

        Parameters
        ----------
        group
            The group within the original Zarr store to be used as the root group for the
            ManifestStore (default: the Zarr store's root group).
        skip_variables
            Variables in the Zarr store that will be ignored when creating the ManifestStore
            (default: `None`, do not ignore any variables).
        """

        self.group = group
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given Zarr store to produce a VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL to the input Zarr store (e.g., "s3://bucket/store.zarr").
        registry
            An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a virtual Zarr representation of the parsed data source.
        """

        path = validate_and_normalize_path_to_uri(url, fs_root=Path.cwd().as_uri())
        import asyncio

        object_store, _ = registry.resolve(path)
        zarr_store = ObjectStore(store=object_store)
        manifest_group = asyncio.run(
            _construct_manifest_group(
                store=zarr_store,
                path=url,
                group=self.group,
                skip_variables=self.skip_variables,
            )
        )
        return ManifestStore(registry=registry, group=manifest_group)
