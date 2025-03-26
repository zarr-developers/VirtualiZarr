from __future__ import annotations

import asyncio
from itertools import starmap
from pathlib import Path  # noqa
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
)

from xarray import Dataset, Index, Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri  # noqa
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.readers.common import (
    construct_fully_virtual_dataset,
    replace_virtual_with_loadable_vars,
)
from virtualizarr.zarr import ZARR_DEFAULT_FILL_VALUE

if TYPE_CHECKING:
    import zarr

# Vendored directly from Zarr-python V3's private API
# https://github.com/zarr-developers/zarr-python/blob/458299857141a5470ba3956d8a1607f52ac33857/src/zarr/core/common.py#L53
T = TypeVar("T", bound=tuple[Any, ...])
V = TypeVar("V")


async def _concurrent_map(
    items: Iterable[T],
    func: Callable[..., Awaitable[V]],
    limit: int | None = None,
) -> list[V]:
    if limit is None:
        return await asyncio.gather(*list(starmap(func, items)))

    else:
        sem = asyncio.Semaphore(limit)

        async def run(item: tuple[Any]) -> V:
            async with sem:
                return await func(*item)

        return await asyncio.gather(
            *[asyncio.ensure_future(run(item)) for item in items]
        )


async def get_chunk_mapping_prefix(
    zarr_array: zarr.AsyncArray, prefix: str, filepath: str
) -> dict:
    """Create a dictionary to pass into ChunkManifest __init__"""

    # TODO: For when we want to support reading V2 we should parse the /c/ and "/" between chunks
    prefix = zarr_array.name + "/c/"
    prefix_keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]
    _lengths = await _concurrent_map(prefix_keys, zarr_array.store.getsize)
    chunk_keys = [x[0].split(prefix)[1] for x in prefix_keys]
    _dict_keys = [key.replace("/", ".") for key in chunk_keys]
    _paths = [filepath + prefix + key for key in chunk_keys]

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
    chunk_map = await get_chunk_mapping_prefix(
        zarr_array, prefix=f"{zarr_array.name}/c", filepath=filepath
    )
    return ChunkManifest(chunk_map)


async def build_zarray_metadata(zarr_array: zarr.AsyncArray[Any]):
    attrs = zarr_array.metadata.attributes

    fill_value = zarr_array.metadata.fill_value
    if fill_value is not None:
        fill_value = ZARR_DEFAULT_FILL_VALUE[zarr_array.metadata.fill_value.dtype.kind]

    zarr_format = zarr_array.metadata.zarr_format

    if zarr_format == 2:
        # TODO: Add ability to read Zarr V2 stores.
        # TODO: Convert Zarr v2 compressors and filters to Zarr v3 compliant codec chain.
        # array_dims = attrs["_ARRAY_DIMENSIONS"]
        raise NotImplementedError("Reading Zarr V2 currently not supported.")

    elif zarr_format == 3:
        array_dims = zarr_array.metadata.dimension_names  # type: ignore[union-attr]
        codec_list = [codec.to_dict() for codec in zarr_array.codec_pipeline]
        if fill_value is None:
            raise ValueError(
                "fill_value must be specified https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value"
            )

    else:
        raise NotImplementedError("Zarr format is not recognized as v2 or v3.")

    # TODO: / Question: We can get a `zarr.core.codec_pipeline.BatchedCodecPipeline`` from zarr-python,
    # which is the end result dtype from create_v3_array_metadata -> convert_to_codec_pipeline.
    # This deconstruction, then reconstuction seems kind of excessive. Should/could we bypass this?
    array_v3_metadata = create_v3_array_metadata(
        shape=zarr_array.shape,
        data_type=zarr_array.dtype.name,
        chunk_shape=zarr_array.chunks,
        fill_value=fill_value,
        codecs=codec_list,
        dimension_names=array_dims,
        attributes=attrs,
    )

    return {
        "zarray_array": array_v3_metadata,
        "array_dims": array_dims,
        "array_metadata": attrs,
    }


async def virtual_variable_from_zarr_array(
    zarr_array: zarr.AsyncArray[Any], filepath: str
):
    zarray_array_dict = await build_zarray_metadata(zarr_array=zarr_array)

    chunk_manifest = await build_chunk_manifest(zarr_array, filepath=filepath)
    manifest_array = ManifestArray(
        metadata=zarray_array_dict["zarray_array"], chunkmanifest=chunk_manifest
    )
    return Variable(
        dims=zarray_array_dict["array_dims"],
        data=manifest_array,
        attrs=zarray_array_dict["array_metadata"],
    )


async def virtual_dataset_from_zarr_group(
    zarr_group: zarr.AsyncGroup,
    filepath: str,
    group: str,
    drop_variables: Iterable[str] | None = [],
    loadable_variables: Iterable[str] | None = [],
    decode_times: bool | None = None,
    indexes: Mapping[str, Index] | None = None,
    reader_options: dict = {},
):
    zarr_array_keys = [key async for key in zarr_group.array_keys()]

    if loadable_variables is None:
        loadable_variables = []

    virtual_zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(var) for var in zarr_array_keys]
    )

    # Xarray Variable backed by manifest array
    virtual_variable_arrays = await asyncio.gather(
        *[
            virtual_variable_from_zarr_array(zarr_array=array, filepath=filepath)  # type: ignore[arg-type]
            for array in virtual_zarr_arrays
        ]
    )

    # build a dict mapping for use later in construct_virtual_dataset
    virtual_variable_array_mapping = {
        array.basename: result
        for array, result in zip(virtual_zarr_arrays, virtual_variable_arrays)
    }

    # flatten nested tuples and get set -> list
    coord_names = list(
        set(
            [
                item
                for tup in [val.dims for val in virtual_variable_arrays]
                for item in tup
            ]
        )
    )

    fully_virtual_dataset = construct_fully_virtual_dataset(
        virtual_vars=virtual_variable_array_mapping,
        coord_names=coord_names,
        attrs=zarr_group.attrs,
    )

    vds = replace_virtual_with_loadable_vars(
        fully_virtual_dataset,
        filepath,
        group=group,
        loadable_variables=loadable_variables,
        reader_options=reader_options,
        indexes=indexes,
        decode_times=decode_times,
    )

    return vds.drop_vars(drop_variables)


class ZarrVirtualBackend(VirtualBackend):
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
        import asyncio

        import zarr

        async def _open_virtual_dataset(
            filepath=filepath,
            group=group,
            drop_variables=drop_variables,
            loadable_variables=loadable_variables,
            decode_times=decode_times,
            indexes=indexes,
            virtual_backend_kwargs=virtual_backend_kwargs,
            reader_options=reader_options,
        ):
            if virtual_backend_kwargs:
                raise NotImplementedError(
                    "Zarr reader does not understand any virtual_backend_kwargs"
                )
            _drop_vars: list[Hashable] = (
                [] if drop_variables is None else list(drop_variables)
            )

            filepath = validate_and_normalize_path_to_uri(
                filepath, fs_root=Path.cwd().as_uri()
            )

            if reader_options is None:
                reader_options = {}

            zg = await zarr.api.asynchronous.open_group(
                filepath,
                storage_options=reader_options.get("storage_options"),
                mode="r",
            )

            return await virtual_dataset_from_zarr_group(
                zarr_group=zg,
                filepath=filepath,
                group=group,
                drop_variables=_drop_vars,
                loadable_variables=loadable_variables,
                decode_times=decode_times,
                indexes=indexes,
                reader_options=reader_options,
            )

        return asyncio.run(_open_virtual_dataset())
