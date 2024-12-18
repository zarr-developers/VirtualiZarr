import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numcodecs
import numpy as np
from xarray import Dataset, Index, Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.readers.common import VirtualBackend, separate_coords
from virtualizarr.zarr import ZArray


class ZarrV3VirtualBackend(VirtualBackend):
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
                "Zarr_v3 reader does not understand any virtual_backend_kwargs"
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
