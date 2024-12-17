from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Optional

import numcodecs
import numpy as np
from xarray import Dataset, Index, Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import (
    validate_and_normalize_path_to_uri,
)
from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
    open_loadable_vars_and_indexes,
    separate_coords,
)
from virtualizarr.utils import check_for_collisions
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    import upath
    import zarr


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
        # Question: Is this something we want to pass through?
        if virtual_backend_kwargs:
            raise NotImplementedError(
                "Zarr reader does not understand any virtual_backend_kwargs"
            )

        import zarr
        from packaging import version

        if version.parse(zarr.__version__).major < 3:
            raise ImportError("Zarr V3 is required")

        drop_variables, loadable_variables = check_for_collisions(
            drop_variables,
            loadable_variables,
        )

        return virtual_dataset_from_zarr_group(
            filepath=filepath,
            group=group,
            drop_variables=drop_variables,
            loadable_variables=loadable_variables,
            decode_times=decode_times,
            indexes=indexes,
            reader_options=reader_options,
        )


class ZarrV3ChunkManifestVirtualBackend(VirtualBackend):
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
        Read a Zarr v3 store containing chunk manifests and return an xarray Dataset containing virtualized arrays.

        This is experimental - chunk manifests are not part of the Zarr v3 Spec.
        """
        if virtual_backend_kwargs:
            raise NotImplementedError(
                "Zarr V3 Chunk Manifest reader does not understand any virtual_backend_kwargs"
            )
        storepath = Path(filepath)

        if group:
            raise NotImplementedError()

        if loadable_variables or decode_times:
            raise NotImplementedError()

        if reader_options:
            raise NotImplementedError()

        drop_vars: list[str]
        if drop_variables is None:
            drop_vars = []
        else:
            drop_vars = list(drop_variables)

        ds_attrs = attrs_from_zarr_group_json(storepath / "zarr.json")
        coord_names = ds_attrs.pop("coordinates", [])

        # TODO recursive glob to create a datatree
        # Note: this .is_file() check should not be necessary according to the pathlib docs, but tests fail on github CI without it
        # see https://github.com/TomNicholas/VirtualiZarr/pull/45#discussion_r1547833166
        all_paths = storepath.glob("*/")
        directory_paths = [p for p in all_paths if not p.is_file()]

        vars = {}
        for array_dir in directory_paths:
            var_name = array_dir.name
            if var_name in drop_vars:
                break

            zarray, dim_names, attrs = metadata_from_zarr_json(array_dir / "zarr.json")
            manifest = ChunkManifest.from_zarr_json(str(array_dir / "manifest.json"))

            marr = ManifestArray(chunkmanifest=manifest, zarray=zarray)
            var = Variable(data=marr, dims=dim_names, attrs=attrs)
            vars[var_name] = var

        if indexes is None:
            raise NotImplementedError()
        elif indexes != {}:
            # TODO allow manual specification of index objects
            raise NotImplementedError()
        else:
            indexes = dict(**indexes)  # for type hinting: to allow mutation

        data_vars, coords = separate_coords(vars, indexes, coord_names)

        ds = Dataset(
            data_vars,
            coords=coords,
            # indexes={},  # TODO should be added in a later version of xarray
            attrs=ds_attrs,
        )

        return ds


def virtual_dataset_from_zarr_group(
    filepath: str,
    group: str | None = None,
    drop_variables: Iterable[str] | None = [],
    loadable_variables: Iterable[str] | None = [],
    decode_times: bool | None = None,
    indexes: Mapping[str, Index] | None = None,
    reader_options: Optional[dict] = None,
) -> Dataset:
    import zarr
    from upath import UPath

    filepath = validate_and_normalize_path_to_uri(filepath, fs_root=Path.cwd().as_uri())
    # This currently fails for local filepaths (ie. tests):
    # *** TypeError: Filesystem needs to support async operations.
    # https://github.com/zarr-developers/zarr-python/issues/2554

    # use UPath for combining store path + chunk key when building chunk manifests
    store_path = UPath(filepath)

    if reader_options is None:
        reader_options = {}

    zg = zarr.open_group(
        filepath, storage_options=reader_options.get("storage_options"), mode="r"
    )

    zarr_arrays = [val for val in zg.keys()]

    # mypy typing
    if loadable_variables is None:
        loadable_variables = set()

    if drop_variables is None:
        drop_variables = set()

    missing_vars = set(loadable_variables) - set(zarr_arrays)
    if missing_vars:
        raise ValueError(
            f"Some loadable variables specified are not present in this zarr store: {missing_vars}"
        )

    virtual_vars = list(
        set(zarr_arrays) - set(loadable_variables) - set(drop_variables)
    )

    virtual_variable_mapping = {
        f"{var}": construct_virtual_array(
            zarr_group=zg, var_name=var, store_path=store_path
        )
        for var in virtual_vars
    }

    coord_names = list(
        set(
            item
            for tup in [
                virtual_variable_mapping[val].dims for val in virtual_variable_mapping
            ]
            for item in tup
        )
    )

    non_loadable_variables = list(set(virtual_vars).union(set(drop_variables)))

    loadable_vars, indexes = open_loadable_vars_and_indexes(
        filepath,
        loadable_variables=loadable_variables,
        reader_options=reader_options,
        drop_variables=non_loadable_variables,
        indexes=indexes,
        group=group,
        decode_times=decode_times,
    )

    return construct_virtual_dataset(
        virtual_vars=virtual_variable_mapping,
        loadable_vars=loadable_vars,
        indexes=indexes,
        coord_names=coord_names,
        attrs=zg.attrs.asdict(),
    )


async def get_chunk_size(zarr_group: zarr.Group, chunk_key: str) -> int:
    # User zarr-pythons `getsize` method to get bytes per chunk
    return await zarr_group.store.getsize(chunk_key)


async def chunk_exists(zarr_group: zarr.Group, chunk_key: str) -> bool:
    # calls zarr-pythons `exists` to check for a chunk
    return await zarr_group.store.exists(chunk_key)


async def list_store_keys(zarr_group: zarr.Group) -> list[str]:
    # Lists all keys in a store
    return [item async for item in zarr_group.store.list()]


async def get_chunk_paths(
    zarr_group: zarr.Group, array_name: str, store_path: upath.core.UPath
) -> dict:
    chunk_paths = {}

    # Can we call list() on an array instead of the entire store?
    store_keys = zarr.core.sync.sync(list_store_keys(zarr_group))

    for item in store_keys:
        # should we move these filters/checks into list_store_keys?
        if (
            not item.endswith((".zarray", ".zattrs", ".zgroup", ".zmetadata", ".json"))
            and item.startswith(array_name)
            and zarr.core.sync.sync(chunk_exists(zarr_group=zarr_group, chunk_key=item))
        ):
            if zarr_group.metadata.zarr_format == 2:
                # split on array name + trailing slash
                chunk_key = item.split(array_name + "/")[-1]

            elif zarr_group.metadata.zarr_format == 3:
                # In v3 we remove the /c/ 'chunks' part of the key and
                # replace trailing slashes with '.' to conform to ChunkManifest validation
                chunk_key = (
                    item.split(array_name + "/")[-1].split("c/")[-1].replace("/", ".")
                )

            else:
                raise NotImplementedError(
                    f"{zarr_group.metadata.zarr_format} not 2 or 3."
                )

            # Can we ask Zarr-python for the path and protocol?
            # This gives path, but no protocol: zarr_group.store.path

            chunk_paths[chunk_key] = {
                "path": ((store_path / item).as_uri()),
                "offset": 0,
                "length": zarr.core.sync.sync(get_chunk_size(zarr_group, item)),
            }

            # Note: This won't work for sharded stores: https://github.com/zarr-developers/VirtualiZarr/pull/271#discussion_r1844487578

    return chunk_paths


def construct_virtual_array(
    zarr_group: zarr.Group, var_name: str, store_path: upath.core.UPath
):
    zarr_array = zarr_group[var_name]

    attrs = zarr_array.metadata.attributes

    if zarr_array.metadata.zarr_format == 2:
        array_zarray = _parse_zarr_v2_metadata(zarr_array=zarr_array)  # type: ignore[arg-type]
        array_dims = attrs["_ARRAY_DIMENSIONS"]

    elif zarr_array.metadata.zarr_format == 3:
        array_zarray = _parse_zarr_v3_metadata(zarr_array=zarr_array)  # type: ignore[arg-type]
        array_dims = zarr_array.metadata.dimension_names  # type: ignore[union-attr]

    else:
        raise NotImplementedError("Zarr format is not recognized as v2 or v3.")

    import asyncio

    array_chunk_sizes = asyncio.run(
        get_chunk_paths(
            zarr_group=zarr_group, array_name=var_name, store_path=store_path
        )
    )

    array_chunkmanifest = ChunkManifest(array_chunk_sizes)

    array_manifest_array = ManifestArray(
        zarray=array_zarray, chunkmanifest=array_chunkmanifest
    )

    array_variable = Variable(
        dims=array_dims,
        data=array_manifest_array,
        attrs=attrs,
    )

    return array_variable


def _parse_zarr_v2_metadata(zarr_array: zarr.Array) -> ZArray:
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


def _parse_zarr_v3_metadata(zarr_array: zarr.Array) -> ZArray:
    from virtualizarr.codecs import get_codecs

    if zarr_array.metadata.fill_value is None:
        raise ValueError(
            "fill_value must be specified https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value"
        )
    else:
        fill_value = zarr_array.metadata.fill_value

    # Codecs from test looks like: (BytesCodec(endian=<Endian.little: 'little'>),)
    # Questions: What do we do with endian info?
    codecs = get_codecs(zarr_array)

    # Question: How should we parse the values from get_codecs?
    # typing: Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]
    # mypy:  ... is not indexable  [index]
    # added tmp bypyass for mypy
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


def attrs_from_zarr_group_json(filepath: Path) -> dict:
    with open(filepath) as metadata_file:
        attrs = json.load(metadata_file)
    return attrs["attributes"]


def metadata_from_zarr_json(filepath: Path) -> tuple[ZArray, list[str], dict]:
    with open(filepath) as metadata_file:
        metadata = json.load(metadata_file)

    if {
        "name": "chunk-manifest-json",
        "configuration": {
            "manifest": "./manifest.json",
        },
    } not in metadata.get("storage_transformers", []):
        raise ValueError(
            "Can only read byte ranges from Zarr v3 stores which implement the manifest storage transformer ZEP."
        )

    attrs = metadata.pop("attributes")
    dim_names = metadata.pop("dimension_names")

    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    shape = tuple(metadata["shape"])
    zarr_format = metadata["zarr_format"]

    if metadata["fill_value"] is None:
        raise ValueError(
            "fill_value must be specified https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value"
        )
    else:
        fill_value = metadata["fill_value"]

    all_codecs = [
        codec
        for codec in metadata["codecs"]
        if codec["name"] not in ("transpose", "bytes")
    ]
    compressor, *filters = [
        _configurable_to_num_codec_config(_filter) for _filter in all_codecs
    ]
    zarray = ZArray(
        chunks=chunk_shape,
        compressor=compressor,
        dtype=np.dtype(metadata["data_type"]),
        fill_value=fill_value,
        filters=filters or None,
        order="C",
        shape=shape,
        zarr_format=zarr_format,
    )

    return zarray, dim_names, attrs


def _configurable_to_num_codec_config(configurable: dict) -> dict:
    """
    Convert a zarr v3 configurable into a numcodecs codec.
    """
    configurable_copy = configurable.copy()
    codec_id = configurable_copy.pop("name")
    if codec_id.startswith("numcodecs."):
        codec_id = codec_id[len("numcodecs.") :]
    configuration = configurable_copy.pop("configuration")

    return numcodecs.get_codec({"id": codec_id, **configuration}).get_config()
