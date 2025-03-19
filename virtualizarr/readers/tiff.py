from __future__ import annotations

import dataclasses
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Optional,
)

import numcodecs.registry as registry
from zarr.core.sync import sync

from virtualizarr.codecs import numcodec_config_to_configurable
from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.storage.obstore import ManifestStore

if TYPE_CHECKING:
    from async_tiff import TIFF
    from async_tiff._ifd import ImageFileDirectory
    from obstore.store import ObjectStore
    from zarr.core.abc.store import Store


import numpy as np
import xarray as xr

from virtualizarr.readers.common import (
    VirtualBackend,
    ZlibProperties,
)


def _get_dtype(sample_format, bits_per_sample):
    if sample_format[0] == 1 and bits_per_sample[0] == 16:
        return np.dtype(np.uint16)
    else:
        raise NotImplementedError


def _get_codecs(compression):
    if compression == 8:  # Adobe DEFLATE
        zlib_props = ZlibProperties(level=6)  # type: ignore
        conf = dataclasses.asdict(zlib_props)
        conf["id"] = "zlib"
    else:
        raise NotImplementedError
    codec = registry.get_codec(conf)
    return codec


class TIFFVirtualBackend(VirtualBackend):
    @staticmethod
    def _construct_chunk_manifest(
        ifd: ImageFileDirectory,
        *,
        path: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
    ) -> ChunkManifest:
        tile_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
        # See https://web.archive.org/web/20240329145228/https://www.awaresystems.be/imaging/tiff/tifftags/tileoffsets.html for ordering of offsets.
        tile_offsets = np.array(ifd.tile_offsets, dtype=np.uint64).reshape(tile_shape)
        tile_counts = np.array(ifd.tile_byte_counts, dtype=np.uint64).reshape(
            tile_shape
        )
        paths = np.full_like(tile_offsets, path, dtype=np.dtypes.StringDType)
        return ChunkManifest.from_arrays(
            paths=paths,
            offsets=tile_offsets,
            lengths=tile_counts,
        )

    @staticmethod
    async def _open_tiff(*, path: str, store: ObjectStore) -> TIFF:
        return await TIFF.open(path, store=store)

    @staticmethod
    def _construct_manifest_array(
        *, ifd: ImageFileDirectory, path: str
    ) -> ManifestArray:
        if not ifd.tile_height or not ifd.tile_width:
            raise NotImplementedError(
                f"TIFF reader currently only supports tiled TIFFs, but {path} has no internal tiling."
            )
        chunks = (ifd.tile_height, ifd.tile_height)
        shape = (ifd.image_height, ifd.image_width)
        chunk_manifest = TIFFVirtualBackend._construct_chunk_manifest(
            ifd, path=path, shape=shape, chunks=chunks
        )
        codecs = [_get_codecs(ifd.compression)]
        codec_configs = [
            numcodec_config_to_configurable(codec.get_config()) for codec in codecs
        ]
        dimension_names = ("y", "x")  # Folllowing rioxarray's behavior

        metadata = create_v3_array_metadata(
            shape=shape,
            data_type=_get_dtype(
                sample_format=ifd.sample_format, bits_per_sample=ifd.bits_per_sample
            ),
            chunk_shape=chunks,
            fill_value=None,  # TODO: Fix fill value
            codecs=codec_configs,
            dimension_names=dimension_names,
        )
        return ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)

    @staticmethod
    def _construct_manifest_group(
        store: ObjectStore,
        path: str,
        *,
        group: str | None = None,
    ) -> ManifestGroup:
        """
        Construct a virtual Group from a tiff file.
        """
        # TODO: Make an async approach
        tiff = sync(TIFFVirtualBackend._open_tiff(store=store, path=path))
        attrs: dict[str, Any] = {}
        manifest_dict = {}
        if group:
            manifest_dict[group] = TIFFVirtualBackend._construct_manifest_array(
                ifd=tiff.ifds[int(group)], path=path
            )
        else:
            for ind, ifd in enumerate(tiff.ifds):
                manifest_dict[str(ind)] = TIFFVirtualBackend._construct_manifest_array(
                    ifd=ifd, path=path
                )
        return ManifestGroup(manifest_dict=manifest_dict, attributes=attrs)

    @staticmethod
    def open_virtual_zarr(
        filepath: str,
        group: str,
        file_id: str,
        object_store: ObjectStore,
    ) -> Store:
        # TODO: Make this less sketchy, but it's better to AsyncTIFF store rather than an obstore store so here we are
        from async_tiff.store import LocalStore as ATStore

        newargs = object_store.__getnewargs_ex__()
        at_store = ATStore(*newargs[0], **newargs[1])

        # Create a group containing dataset level metadata and all the manifest arrays
        manifest_group = TIFFVirtualBackend._construct_manifest_group(
            store=at_store, path=filepath, group=group
        )
        # Convert to a manifest store
        return ManifestStore(
            stores={file_id: object_store}, manifest_group=manifest_group
        )

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
        if not group:
            raise ValueError("Group is required for the TIFFVirtualBackend")
        if not virtual_backend_kwargs:
            raise ValueError(
                "virtual_backend_kwargs is required for specifying the store and file_id"
            )
        store = virtual_backend_kwargs.get("store", None)
        if not store:
            raise ValueError(
                f"Expected 'store' key in virtual_backend_kwargs, got {virtual_backend_kwargs}"
            )
        file_id = virtual_backend_kwargs.get("file_id", None)
        if not file_id:
            raise ValueError(
                f"Expected 'file_id' key in virtual_backend_kwargs, got {virtual_backend_kwargs}"
            )
        manifest_store = TIFFVirtualBackend.open_virtual_zarr(
            filepath=filepath,
            group=group,
            object_store=store,
            file_id=file_id,
        )

        # Convert to a Zarr store
        return xr.open_dataset(
            manifest_store, engine="zarr", consolidated=False, zarr_format=3
        )
