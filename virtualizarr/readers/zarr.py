from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

from xarray import Dataset, Index, Variable
from zarr.core.common import concurrent_map

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
    maybe_open_loadable_vars_and_indexes,
)
from virtualizarr.utils import check_for_collisions
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    import zarr


async def _parse_zarr_v2_metadata(zarr_array: zarr.Array) -> ZArray:
    return ZArray(
        shape=zarr_array.metadata.shape,
        chunks=zarr_array.metadata.chunks,  # type: ignore[union-attr]
        dtype=zarr_array.metadata.dtype,
        fill_value=zarr_array.metadata.fill_value,  # type: ignore[arg-type]
        order="C",
        compressor=zarr_array.metadata.compressor,  # type: ignore[union-attr]
        filters=zarr_array.metadata.filters,  # type: ignore
        zarr_format=zarr_array.metadata.zarr_format,
    )


async def _parse_zarr_v3_metadata(zarr_array: zarr.Array) -> ZArray:
    from virtualizarr.codecs import get_codecs

    if zarr_array.metadata.fill_value is None:
        raise ValueError(
            "fill_value must be specified https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value"
        )
    else:
        fill_value = zarr_array.metadata.fill_value

    codecs = get_codecs(zarr_array)

    # Question: How should we parse the values from get_codecs?
    compressor = getattr(codecs[0], "compressor", None)  # type: ignore
    filters = getattr(codecs[0], "filters", None)  # type: ignore

    return ZArray(
        chunks=zarr_array.metadata.chunk_grid.chunk_shape,  # type: ignore[attr-defined]
        compressor=compressor,
        dtype=zarr_array.metadata.data_type.name,  # type: ignore
        fill_value=fill_value,  # type: ignore[arg-type]
        filters=filters,
        order="C",
        shape=zarr_array.metadata.shape,
        zarr_format=zarr_array.metadata.zarr_format,
    )


async def build_chunk_manifest_from_dict_mapping(
    store_path: str, chunk_mapping_dict: dict, array_name: str, zarr_format: int
) -> ChunkManifest:
    chunk_manifest_dict = {}
    # ToDo: We could skip the dict creation and built the Manifest from arrays directly
    for key, value in chunk_mapping_dict.items():
        if zarr_format == 2:
            # split on array name + trailing slash
            chunk_key = key.split(array_name + "/")[-1]

        elif zarr_format == 3:
            # In v3 we remove the /c/ 'chunks' part of the key and
            # replace trailing slashes with '.' to conform to ChunkManifest validation
            chunk_key = (
                key.split(array_name + "/")[-1].split("c/")[-1].replace("/", ".")
            )

        chunk_manifest_dict[chunk_key] = {
            "path": store_path + "/" + key,
            "offset": 0,
            "length": value,
        }

    return ChunkManifest(chunk_manifest_dict)


async def build_chunk_manifest(
    zarr_array: zarr.AsyncArray, prefix: str, filepath: str
) -> ChunkManifest:
    """Build a ChunkManifest with the from_arrays method"""
    import numpy as np

    # import pdb; pdb.set_trace()
    key_tuples = [(x,) async for x in zarr_array.store.list_prefix(prefix)]

    filepath_list = [filepath] * len(key_tuples)

    # can this be lambda'ed?
    # stolen from manifest.py
    def combine_path_chunk(filepath: str, chunk_key: str):
        return filepath + "/" + chunk_key

    vectorized_chunk_path_combine_func = np.vectorize(
        combine_path_chunk, otypes=[np.dtypes.StringDType()]
    )

    # _paths: np.ndarray[Any, np.dtypes.StringDType]
    # turn the tuples of chunks to a flattened list with :list(sum(key_tuples, ()))
    _paths = vectorized_chunk_path_combine_func(
        filepath_list, list(sum(key_tuples, ()))
    )

    # _offsets: np.ndarray[Any, np.dtype[np.uint64]]
    _offsets = np.array([0] * len(_paths), dtype=np.uint64)

    # _lengths: np.ndarray[Any, np.dtype[np.uint64]]
    lengths = await concurrent_map((key_tuples), zarr_array.store.getsize)
    _lengths = np.array(lengths, dtype=np.uint64)

    return ChunkManifest.from_arrays(
        paths=_paths,  # type: ignore
        offsets=_offsets,
        lengths=_lengths,
    )


async def get_chunk_mapping_prefix(zarr_array: zarr.AsyncArray, prefix: str) -> dict:
    """Create a chunk map"""

    keys = [(x,) async for x in zarr_array.store.list_prefix(prefix)]

    sizes = await concurrent_map(keys, zarr_array.store.getsize)
    return {key[0]: size for key, size in zip(keys, sizes)}


async def build_zarray_metadata(zarr_array: zarr.AsyncArray):
    attrs = zarr_array.metadata.attributes

    if zarr_array.metadata.zarr_format == 2:
        array_zarray = await _parse_zarr_v2_metadata(zarr_array=zarr_array)  # type: ignore[arg-type]
        array_dims = attrs["_ARRAY_DIMENSIONS"]

    elif zarr_array.metadata.zarr_format == 3:
        array_zarray = await _parse_zarr_v3_metadata(zarr_array=zarr_array)  # type: ignore[arg-type]
        array_dims = zarr_array.metadata.dimension_names  # type: ignore[union-attr]

    else:
        raise NotImplementedError("Zarr format is not recognized as v2 or v3.")

    return {
        "zarray_array": array_zarray,
        "array_dims": array_dims,
        "array_metadata": attrs,
    }


async def virtual_variable_from_zarr_array(zarr_array: zarr.AsyncArray, filepath: str):
    zarr_prefix = zarr_array.basename

    if zarr_array.metadata.zarr_format == 3:
        # if we have Zarr format/version 3, we add /c/ to the chunk paths
        zarr_prefix = f"{zarr_prefix}/c"

    zarray_array = await build_zarray_metadata(zarr_array=zarr_array)

    chunk_manifest = await build_chunk_manifest(
        zarr_array, prefix=zarr_prefix, filepath=filepath
    )

    manifest_array = ManifestArray(
        zarray=zarray_array["zarray_array"], chunkmanifest=chunk_manifest
    )
    return Variable(
        dims=zarray_array["array_dims"],
        data=manifest_array,
        attrs=zarray_array["array_metadata"],
    )


async def virtual_dataset_from_zarr_group(
    zarr_group: zarr.AsyncGroup,
    filepath: str,
    group: str,
    drop_variables: Iterable[str] | None = [],
    virtual_variables: Iterable[str] | None = [],
    loadable_variables: Iterable[str] | None = [],
    decode_times: bool | None = None,
    indexes: Mapping[str, Index] | None = None,
    reader_options: dict = {},
):
    virtual_zarr_arrays = await asyncio.gather(
        *[zarr_group.getitem(var) for var in virtual_variables]
    )

    virtual_variable_arrays = await asyncio.gather(
        *[
            virtual_variable_from_zarr_array(zarr_array=array, filepath=filepath)
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

    non_loadable_variables = list(set(virtual_variables).union(set(drop_variables)))

    loadable_vars, indexes = maybe_open_loadable_vars_and_indexes(
        filepath,
        loadable_variables=loadable_variables,
        reader_options=reader_options,
        drop_variables=non_loadable_variables,
        indexes=indexes,
        group=group,
        decode_times=decode_times,
    )

    return construct_virtual_dataset(
        virtual_vars=virtual_variable_array_mapping,
        loadable_vars=loadable_vars,
        indexes=indexes,
        coord_names=coord_names,
        attrs=zarr_group.attrs,
    )


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
        from packaging import version

        if version.parse(zarr.__version__).major < 3:
            raise ImportError("Zarr V3 is required")

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

            drop_variables, loadable_variables = check_for_collisions(
                drop_variables,
                loadable_variables,
            )

            filepath = validate_and_normalize_path_to_uri(
                filepath, fs_root=Path.cwd().as_uri()
            )
            # This currently fails for local filepaths (ie. tests) but works for s3:
            # *** TypeError: Filesystem needs to support async operations.
            # https://github.com/zarr-developers/zarr-python/issues/2554

            if reader_options is None:
                reader_options = {}

            zg = await zarr.api.asynchronous.open_group(
                filepath,
                storage_options=reader_options.get("storage_options"),
                mode="r",
            )

            zarr_array_keys = [key async for key in zg.array_keys()]

            missing_vars = set(loadable_variables) - set(zarr_array_keys)
            if missing_vars:
                raise ValueError(
                    f"Some loadable variables specified are not present in this zarr store: {missing_vars}"
                )
            virtual_vars = list(
                set(zarr_array_keys) - set(loadable_variables) - set(drop_variables)
            )

            return await virtual_dataset_from_zarr_group(
                zarr_group=zg,
                filepath=filepath,
                group=group,
                virtual_variables=virtual_vars,
                drop_variables=drop_variables,
                loadable_variables=loadable_variables,
                decode_times=decode_times,
                indexes=indexes,
                reader_options=reader_options,
            )

        # How does this asyncio.run call interact with zarr-pythons async event loop?
        return asyncio.run(_open_virtual_dataset())
