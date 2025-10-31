from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path  # noqa
from typing import TYPE_CHECKING, Any, Hashable

import zarr
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

if TYPE_CHECKING:
    import zarr

ZarrArrayType = zarr.AsyncArray | zarr.Array


async def get_chunk_mapping_prefix(zarr_array: ZarrArrayType, path: str) -> dict:
    """Create a dictionary to pass into ChunkManifest __init__"""

    zarr_format = zarr_array.metadata.zarr_format
    
    if zarr_format == 2:
        # V2 chunk paths don't have /c/ prefix
        # They're like "array_name/0.0.0" or "array_name/0"
        prefix = zarr_array.name.lstrip("/") + "/"
        
        if zarr_array.shape == ():
            # V2 scalar arrays have a single chunk at "0"
            chunk_key = "0"
            size = await zarr_array.store.getsize(prefix + chunk_key)
            return {"0": {"path": path + "/" + prefix + chunk_key, "offset": 0, "length": size}}
        
        # List all files for the array
        prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
        if not prefix_keys:
            return {}
        
        # Filter out metadata files (.zarray, .zattrs, .zgroup, etc.)
        metadata_files = {'.zarray', '.zattrs', '.zgroup', '.zmetadata'}
        chunk_keys = []
        for key_tuple in prefix_keys:
            key = key_tuple[0]
            # Get the file name after the prefix
            file_name = key.split(prefix)[1] if prefix in key else key.split("/")[-1]
            # Only include if it's not a metadata file
            if file_name not in metadata_files:
                chunk_keys.append(key)
        
        if not chunk_keys:
            return {}
            
        # Now process only actual chunk files
        _lengths = await _concurrent_map([(k,) for k in chunk_keys], zarr_array.store.getsize)
        
        # Extract chunk keys (remove the array name prefix)
        _dict_keys = [key.split(prefix)[1] for key in chunk_keys]
        _paths = [path + "/" + key for key in chunk_keys]
        _offsets = [0] * len(_lengths)
        
        return {
            key: {"path": path, "offset": offset, "length": length}
            for key, path, offset, length in zip(_dict_keys, _paths, _offsets, _lengths)
        }
    
    else:  # V3
        # V3 chunk paths have /c/ prefix
        if zarr_array.shape == ():
            # If we have a scalar array `c`
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
            for key, path, offset, length in zip(_dict_keys, _paths, _offsets, _lengths)
        }


async def build_chunk_manifest(zarr_array: ZarrArrayType, path: str) -> ChunkManifest:
    """Build a ChunkManifest from a dictionary"""
    chunk_map = await get_chunk_mapping_prefix(zarr_array=zarr_array, path=path)

     # If no chunks found (e.g., inline data in V2), provide the shape
    if not chunk_map:
        # Calculate the chunk grid shape from array shape and chunk shape
        import math
        array_shape = zarr_array.shape
        chunk_shape = zarr_array.chunks
        
        if array_shape and chunk_shape:
            chunk_grid_shape = tuple(
                math.ceil(s / c) for s, c in zip(array_shape, chunk_shape)
            )
            return ChunkManifest(chunk_map, shape=chunk_grid_shape)
    
    return ChunkManifest(chunk_map)


def get_metadata(zarr_array: ZarrArrayType) -> ArrayV3Metadata:
    zarr_format = zarr_array.metadata.zarr_format
    if zarr_format == 2:
        # TODO: Once we want to support V2, we will have to deconstruct the
        # zarr_array codecs etc. and reconstruct them with create_v3_array_metadata
    if zarr_format == 2:
        from zarr.metadata.migrate_v3 import _convert_array_metadata
        from zarr.core.metadata import ArrayV2Metadata
        
        # Try standard conversion first
        try:
            v3_metadata = _convert_array_metadata(zarr_array.metadata)
        except TypeError as e:
            # Handle the specific case where V2 has fill_value=None
            if "Cannot convert object None" in str(e) and zarr_array.metadata.fill_value is None:
                # Get the V2 metadata as a dict and update fill_value
                v2_dict = zarr_array.metadata.to_dict()
                v2_dict['fill_value'] = 0  # Temporary value
                
                # Create new V2 metadata with the temporary fill value
                temp_v2 = ArrayV2Metadata.from_dict(v2_dict)
                
                # Convert to V3
                v3_metadata = _convert_array_metadata(temp_v2)
                
                # Get the proper default fill value from the V3 DataType
                default_fill = v3_metadata.data_type.default_scalar()
                
                # Update the V3 metadata with the correct default
                v3_dict = v3_metadata.to_dict()
                v3_dict['fill_value'] = default_fill.item()
                
                # Recreate the V3 metadata with the proper default
                v3_metadata = ArrayV3Metadata.from_dict(v3_dict)
            else:
                raise
        
        # CRITICAL: Ensure dimension_names are set from _ARRAY_DIMENSIONS attribute
        if v3_metadata.dimension_names is None:
            # V2 stores dimension names in the _ARRAY_DIMENSIONS attribute
            if hasattr(zarr_array.metadata, 'attributes') and zarr_array.metadata.attributes:
                dim_names = zarr_array.metadata.attributes.get('_ARRAY_DIMENSIONS', None)
                if dim_names:
                    # Update the V3 metadata with the actual dimension names
                    v3_dict = v3_metadata.to_dict()
                    v3_dict['dimension_names'] = dim_names
                    v3_metadata = ArrayV3Metadata.from_dict(v3_dict)
                else:
                    # Generate unique dimension names based on array name
                    array_name = zarr_array.name.lstrip('/')
                    dim_names = [f'{array_name}_dim_{i}' for i in range(len(zarr_array.shape))]
                    v3_dict = v3_metadata.to_dict()
                    v3_dict['dimension_names'] = dim_names
                    v3_metadata = ArrayV3Metadata.from_dict(v3_dict)
        
        return v3_metadata
    
    elif zarr_format == 3:
        return zarr_array.metadata  # type: ignore[return-value]

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
