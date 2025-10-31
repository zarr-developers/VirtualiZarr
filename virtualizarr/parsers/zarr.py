from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.vendor.zarr.core.common import _concurrent_map

if TYPE_CHECKING:
    import zarr

ZarrArrayType = zarr.AsyncArray | zarr.Array


def join_url(base: str, key: str) -> str:
    """Join a base URL (like s3://bucket/store.zarr) with an object key.

    Ensures we don't accidentally produce double slashes (after the scheme)
    and that the returned string is scheme-friendly.
    """
    if not base:
        return key
    # strip trailing slash from base and leading slash from key to avoid '//' in middle
    return base.rstrip("/") + "/" + key.lstrip("/")


async def get_chunk_mapping_prefix(zarr_array: ZarrArrayType, path: str) -> dict:
    """
    Create a mapping of chunk coordinates to their storage locations.

    Returns a dictionary mapping chunk keys (pure coordinates like "0.1.2")
    to their file paths, offsets, and lengths.
    """
    zarr_format = zarr_array.metadata.zarr_format
    name = getattr(zarr_array, "name", "") or ""
    name = name.lstrip("/")

    if zarr_format == 2:
        prefix = f"{name}/" if name else ""

        if zarr_array.shape == ():
            chunk_key = "0"
            object_key = f"{prefix}{chunk_key}"
            size = await zarr_array.store.getsize(object_key)
            actual_path = join_url(path, object_key)
            return {"0": {"path": actual_path, "offset": 0, "length": size}}

        # List all keys under the array prefix, filtering out metadata files
        prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
        if not prefix_keys:
            return {}

        metadata_files = {".zarray", ".zattrs", ".zgroup", ".zmetadata"}

        metadata_files = {".zarray", ".zattrs", ".zgroup", ".zmetadata"}
        chunk_keys = []
        for key_tuple in prefix_keys:
            key = key_tuple[0]
            file_name = (
                key[len(prefix) :]
                if prefix and key.startswith(prefix)
                else key.split("/")[-1]
            )
            if file_name not in metadata_files:
                chunk_keys.append(key)

        if not chunk_keys:
            return {}

        _lengths = await _concurrent_map(
            [(k,) for k in chunk_keys], zarr_array.store.getsize
        )
        chunk_coords = [
            k[len(prefix) :] if prefix and k.startswith(prefix) else k
            for k in chunk_keys
        ]

        # Normalize to dot-separated coordinates (V2 can use either '/' or '.')
        _dict_keys = [coord.replace("/", ".") for coord in chunk_coords]
        _paths = [join_url(path, k) for k in chunk_keys]
        _offsets = [0] * len(_lengths)

        return {
            key: {"path": p, "offset": offset, "length": length}
            for key, p, offset, length in zip(_dict_keys, _paths, _offsets, _lengths)
        }

    # V3
    else:
        if zarr_array.shape == ():
            prefix = f"{name}/c" if name else "c"
            size = await zarr_array.store.getsize(prefix)
            return {"c": {"path": join_url(path, prefix), "offset": 0, "length": size}}

        prefix = f"{name}/c/" if name else "c/"
        prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
        if not prefix_keys:
            return {}

        _lengths = await _concurrent_map(prefix_keys, zarr_array.store.getsize)
        chunk_keys = [x[0].split(prefix)[1] for x in prefix_keys]
        _dict_keys = [key.replace("/", ".") for key in chunk_keys]
        _paths = [join_url(path, prefix + key) for key in chunk_keys]
        _offsets = [0] * len(_lengths)

        return {
            key: {"path": p, "offset": offset, "length": length}
            for key, p, offset, length in zip(_dict_keys, _paths, _offsets, _lengths)
        }


async def build_chunk_manifest(zarr_array: ZarrArrayType, path: str) -> ChunkManifest:
    """Build a ChunkManifest from chunk coordinate mappings."""
    chunk_map = await get_chunk_mapping_prefix(zarr_array=zarr_array, path=path)

    if not chunk_map:
        import math

        if zarr_array.shape and zarr_array.chunks:
            chunk_grid_shape = tuple(
                math.ceil(s / c) for s, c in zip(zarr_array.shape, zarr_array.chunks)
            )
            return ChunkManifest(chunk_map, shape=chunk_grid_shape)

    return ChunkManifest(chunk_map)


def get_metadata(zarr_array: ZarrArrayType) -> ArrayV3Metadata:
    """
    Get V3 metadata for an array, converting from V2 if necessary.

    For V2 arrays, this performs a complete conversion to V3 metadata including:
    - Converting the metadata structure
    - Handling None fill values
    - Setting dimension names from attributes or generating defaults
    - Replacing V2ChunkKeyEncoding with V3's DefaultChunkKeyEncoding
    """
    zarr_format = zarr_array.metadata.zarr_format

    if zarr_format == 2:
        from zarr.core.metadata import ArrayV2Metadata
        from zarr.metadata.migrate_v3 import _convert_array_metadata

        # Convert V2 metadata to V3
        v2_metadata = zarr_array.metadata
        assert isinstance(v2_metadata, ArrayV2Metadata)

        try:
            v3_metadata = _convert_array_metadata(v2_metadata)
        except TypeError as e:
            # Handle None fill_value case
            if (
                "Cannot convert object None" in str(e)
                and v2_metadata.fill_value is None
            ):
                v2_dict = v2_metadata.to_dict()
                v2_dict["fill_value"] = 0
                temp_v2 = ArrayV2Metadata.from_dict(v2_dict)
                v3_metadata = _convert_array_metadata(temp_v2)

                # Replace with proper default for the data type
                default_scalar = v3_metadata.data_type.default_scalar()
                fill_value = (
                    default_scalar.item()
                    if hasattr(default_scalar, "item")
                    else default_scalar
                )
                v3_dict = v3_metadata.to_dict()
                v3_dict["fill_value"] = fill_value
                v3_metadata = ArrayV3Metadata.from_dict(v3_dict)
            else:
                raise

        # Set dimension names from attributes or generate defaults
        if v3_metadata.dimension_names is None:
            v3_dict = v3_metadata.to_dict()
            if hasattr(v2_metadata, "attributes") and v2_metadata.attributes:
                dim_names = v2_metadata.attributes.get("_ARRAY_DIMENSIONS")
                if dim_names:
                    v3_dict["dimension_names"] = dim_names
                else:
                    array_name = (
                        zarr_array.name.lstrip("/") if zarr_array.name else "array"
                    )
                    v3_dict["dimension_names"] = [
                        f"{array_name}_dim_{i}" for i in range(len(zarr_array.shape))
                    ]
            else:
                array_name = zarr_array.name.lstrip("/") if zarr_array.name else "array"
                v3_dict["dimension_names"] = [
                    f"{array_name}_dim_{i}" for i in range(len(zarr_array.shape))
                ]
            v3_metadata = ArrayV3Metadata.from_dict(v3_dict)

        # CRITICAL: Replace V2ChunkKeyEncoding with V3 DefaultChunkKeyEncoding
        # The automatic conversion preserves V2's encoding, causing zarr to use V2-style
        # paths (array/0) instead of V3-style (array/c/0). This ensures V3 semantics.
        v3_dict = v3_metadata.to_dict()
        v3_dict["chunk_key_encoding"] = {"name": "default", "separator": "."}
        v3_metadata = ArrayV3Metadata.from_dict(v3_dict)

        return v3_metadata

    elif zarr_format == 3:
        return zarr_array.metadata  # type: ignore[return-value]

    else:
        raise NotImplementedError(f"Zarr format {zarr_format} is not supported")


async def _construct_manifest_array(
    zarr_array: zarr.AsyncArray[Any], path: str
) -> ManifestArray:
    """Construct a ManifestArray from a zarr array."""
    array_metadata = get_metadata(zarr_array)
    chunk_manifest = await build_chunk_manifest(zarr_array, path)
    return ManifestArray(metadata=array_metadata, chunkmanifest=chunk_manifest)


async def _construct_manifest_group(
    path: str,
    store: zarr.storage.ObjectStore,
    *,
    skip_variables: str | Iterable[str] | None = None,
    group: str | None = None,
) -> ManifestGroup:
    """Construct a ManifestGroup from a zarr group."""
    zarr_group = await open_group_async(store=store, path=group, mode="r")

    zarr_array_keys = [key async for key in zarr_group.array_keys()]
    _skip_variables = [] if skip_variables is None else list(skip_variables)

    zarr_arrays = await asyncio.gather(
        *[
            zarr_group.getitem(var)
            for var in zarr_array_keys
            if var not in _skip_variables
        ]
    )

    manifest_arrays = await asyncio.gather(
        *[_construct_manifest_array(array, path) for array in zarr_arrays]  # type: ignore[arg-type]
    )

    manifest_dict = {
        array.basename: result for array, result in zip(zarr_arrays, manifest_arrays)
    }

    manifest_group = ManifestGroup(manifest_dict, attributes=zarr_group.attrs)

    # Set V3 group metadata to ensure zarr/xarray uses V3 semantics
    try:
        from zarr.core.group import GroupMetadata

        manifest_group._metadata = GroupMetadata(
            attributes=dict(zarr_group.attrs) if zarr_group.attrs is not None else {},
            zarr_format=3,
            consolidated_metadata=None,
        )
    except Exception:
        pass

    return manifest_group


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
