from __future__ import annotations

import asyncio
from pathlib import Path  # noqa
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterable,
    Mapping,
    Optional,
)

import numpy as np
from xarray import Dataset, Index
from zarr.api.asynchronous import open_group as open_group_async
from zarr.core.metadata import ArrayV3Metadata

from virtualizarr.manifests import (
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri  # noqa
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.vendor.zarr.core.common import _concurrent_map

if TYPE_CHECKING:
    pass

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


import zarr


async def get_chunk_mapping_prefix(zarr_array: zarr.AsyncArray, filepath: str) -> dict:
    """Create a dictionary to pass into ChunkManifest __init__"""

    # TODO: For when we want to support reading V2 we should parse the /c/ and "/" between chunks

    prefix = zarr_array.name.lstrip("/") + "/c/"
    prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
    _lengths = await _concurrent_map(prefix_keys, zarr_array.store.getsize)

    chunk_keys = [x[0].split(prefix)[1] for x in prefix_keys]
    _dict_keys = [key.replace("/", ".") for key in chunk_keys]
    _paths = [filepath + "/" + prefix + key for key in chunk_keys]

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


async def build_chunk_manifest(
    zarr_array: zarr.AsyncArray, filepath: str
) -> ChunkManifest:
    """Build a ChunkManifest from a dictionary"""
    chunk_map = await get_chunk_mapping_prefix(zarr_array=zarr_array, filepath=filepath)
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


async def _construct_manifest_array(zarr_array: zarr.AsyncArray[Any], filepath: str):
    array_metadata = get_metadata(zarr_array=zarr_array)

    chunk_manifest = await build_chunk_manifest(zarr_array, filepath=filepath)
    return ManifestArray(metadata=array_metadata, chunkmanifest=chunk_manifest)


async def _construct_manifest_group(
    filepath: str,
    *,
    reader_options: Optional[dict] = None,
    drop_variables: str | Iterable[str] | None = None,
    group: str | None = None,
):
    reader_options = reader_options or {}
    zarr_group = await open_group_async(
        filepath,
        storage_options=reader_options.get("storage_options"),
        path=group,
        mode="r",
    )

    zarr_array_keys = [key async for key in zarr_group.array_keys()]

    _drop_vars: list[Hashable] = [] if drop_variables is None else list(drop_variables)

    zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(var) for var in zarr_array_keys if var not in _drop_vars]
    )

    manifest_arrays = await asyncio.gather(
        *[
            _construct_manifest_array(zarr_array=array, filepath=filepath)  # type: ignore[arg-type]
            for array in zarr_arrays
        ]
    )

    manifest_dict = {
        array.basename: result for array, result in zip(zarr_arrays, manifest_arrays)
    }
    return ManifestGroup(manifest_dict, attributes=zarr_group.attrs)


def _construct_manifest_store(
    filepath: str,
    *,
    reader_options: Optional[dict] = None,
    drop_variables: str | Iterable[str] | None = None,
    group: str | None = None,
) -> ManifestStore:
    import asyncio

    manifest_group = asyncio.run(
        _construct_manifest_group(
            filepath=filepath,
            group=group,
            drop_variables=drop_variables,
            reader_options=reader_options,
        )
    )
    return ManifestStore(manifest_group)


class ZarrVirtualBackend(VirtualBackend):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: str | Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        filepath = validate_and_normalize_path_to_uri(
            filepath, fs_root=Path.cwd().as_uri()
        )

        manifest_store = _construct_manifest_store(
            filepath=filepath,
            group=group,
            drop_variables=drop_variables,
            reader_options=reader_options,
        )

        ds = manifest_store.to_virtual_dataset(
            loadable_variables=loadable_variables,
            decode_times=decode_times,
            indexes=indexes,
        )
        return ds
