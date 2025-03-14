from __future__ import annotations

import asyncio
import dataclasses
import math
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numcodecs.registry as registry
from async_tiff import TIFF

from virtualizarr.codecs import numcodec_config_to_configurable
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import create_v3_array_metadata

if TYPE_CHECKING:
    from async_tiff._ifd import ImageFileDirectory
    from obstore.store import ObjectStore


import numpy as np
from xarray import DataArray, Dataset, Index, Variable

from virtualizarr.readers.common import (
    VirtualBackend,
    ZlibProperties,
    construct_virtual_dataarray,
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
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        """
        Opens a TIFF with multiple ImageFileDirectories that share a common width and height as an
        Xarray Dataset.
        """
        raise NotImplementedError

    @staticmethod
    def _contruct_chunk_manifest(
        ifd: TIFF.ifds,
        filepath: str,
    ) -> ChunkManifest:
        shape = (ifd.image_height, ifd.image_width)
        chunks = (ifd.tile_height, ifd.tile_height)
        if chunks == (None, None):
            raise NotImplementedError(
                f"TIFF reader currently only supports tiled TIFFs, but {filepath} has no internal tiling."
            )
        tile_shape = tuple(math.ceil(a / b) for a, b in zip(shape, chunks))
        # See https://web.archive.org/web/20240329145228/https://www.awaresystems.be/imaging/tiff/tifftags/tileoffsets.html for ordering of offsets.
        tile_offsets = np.array(ifd.tile_offsets, dtype=np.uint64).reshape(tile_shape)
        tile_counts = np.array(ifd.tile_byte_counts, dtype=np.uint64).reshape(
            tile_shape
        )
        paths = np.full_like(tile_offsets, filepath, dtype=np.dtypes.StringDType)
        return ChunkManifest.from_arrays(
            paths=paths,
            offsets=tile_offsets,
            lengths=tile_counts,
        )

    @staticmethod
    async def _open_tiff(filepath: str, store: ObjectStore) -> TIFF:
        return await TIFF.open(filepath, store=store)

    @staticmethod
    def _contruct_virtual_variable(ifd: ImageFileDirectory, filepath: str) -> Variable:
        chunk_manifest = TIFFVirtualBackend._contruct_chunk_manifest(ifd, filepath)
        codecs = [_get_codecs(ifd.compression)]
        codec_configs = [
            numcodec_config_to_configurable(codec.get_config()) for codec in codecs
        ]
        dimension_names = ["y", "x"]  # Folllowing rioxarray's behavior
        metadata = create_v3_array_metadata(
            shape=(
                ifd.image_height,
                ifd.image_width,
            ),  # TODO: Check if height and width are always ordered along the same axes
            data_type=_get_dtype(
                sample_format=ifd.sample_format, bits_per_sample=ifd.bits_per_sample
            ),
            chunk_shape=(ifd.tile_height, ifd.tile_height),
            fill_value=None,  # TODO: Fix fill value
            codecs=codec_configs,
            dimension_names=dimension_names,
        )
        manifest_array = ManifestArray(metadata=metadata, chunkmanifest=chunk_manifest)
        variable = Variable(data=manifest_array, dims=dimension_names, attrs=None)
        return variable

    @staticmethod
    def open_virtual_dataarray(
        filepath: str,
        *,
        group: str | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> DataArray:
        """
        Opens a TIFF with a single ImageFileDirectory as an Xarray DataArray.
        """
        if group:
            raise NotImplementedError(
                f"Expected None for group, got {group}. Please use the `ifd` keyword to specify which IFD to virtualize."
            )
        if reader_options:
            raise NotImplementedError(
                f"reader_options is not supported for TIFFVirtualBackend but got {reader_options}."
            )
        if not virtual_backend_kwargs:
            raise ValueError(
                f"The store must be supplied to TIFFVirtualBackend through `virtual_backend_kwargs`, but received `virtual_backend_kwargs={virtual_backend_kwargs}`"
            )
        store = virtual_backend_kwargs["store"]
        image_file_directory = virtual_backend_kwargs.get("image_file_directory", 0)
        # TODO: Make an async approach
        tiff = asyncio.run(TIFFVirtualBackend._open_tiff(filepath, store))
        ifd = tiff.ifds[image_file_directory]
        variable = TIFFVirtualBackend._contruct_virtual_variable(
            ifd=ifd, filepath=filepath
        )
        return construct_virtual_dataarray(variable, dims=variable.dims)
