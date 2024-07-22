from typing import List, Tuple, TypedDict, Union

import h5py
import hdf5plugin
import numcodecs.registry as registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs.fixedscaleoffset import FixedScaleOffset
from pydantic import BaseModel, field_validator
from xarray.coding.variables import _choose_float_dtype

_non_standard_filters = {
    "gzip": "zlib",
    "lzf": "imagecodecs_lzf",
}

_hdf5plugin_imagecodecs = {"lz4": "imagecodecs_lz4h5", "bzip2": "imagecodecs_bz2"}


class BloscProperties(BaseModel):
    blocksize: int
    clevel: int
    shuffle: int
    cname: str

    @field_validator("cname", mode="before")
    def get_cname_from_code(cls, v):
        blosc_compressor_codes = {
            value: key
            for key, value in hdf5plugin._filters.Blosc._Blosc__COMPRESSIONS.items()
        }
        return blosc_compressor_codes[v]


class ZstdProperties(BaseModel):
    level: int


class ShuffleProperties(BaseModel):
    elementsize: int


class ZlibProperties(BaseModel):
    level: int


class CFCodec(TypedDict):
    target_dtype: np.dtype
    codec: Codec


def _filter_to_codec(
    filter_id: str, filter_properties: Union[int, None, Tuple] = None
) -> Codec:
    id_int = None
    id_str = None
    try:
        id_int = int(filter_id)
    except ValueError:
        id_str = filter_id
    conf = {}
    if id_str:
        if id_str in _non_standard_filters.keys():
            id = _non_standard_filters[id_str]
        else:
            id = id_str
        if id == "zlib":
            zlib_props = ZlibProperties(level=filter_properties)
            conf = zlib_props.model_dump()  # type: ignore[assignment]
        if id == "shuffle" and isinstance(filter_properties, tuple):
            shuffle_props = ShuffleProperties(elementsize=filter_properties[0])
            conf = shuffle_props.model_dump()  # type: ignore[assignment]
        conf["id"] = id  # type: ignore[assignment]
    if id_int:
        filter = hdf5plugin.get_filters(id_int)[0]
        id = filter.filter_name
        if id in _hdf5plugin_imagecodecs.keys():
            id = _hdf5plugin_imagecodecs[id]
        if id == "blosc" and isinstance(filter_properties, tuple):
            blosc_props = BloscProperties(
                **{
                    k: v
                    for k, v in zip(
                        BloscProperties.model_fields.keys(), filter_properties[-4:]
                    )
                }
            )
            conf = blosc_props.model_dump()  # type: ignore[assignment]
        if id == "zstd" and isinstance(filter_properties, tuple):
            zstd_props = ZstdProperties(level=filter_properties[0])
            conf = zstd_props.model_dump()  # type: ignore[assignment]
        conf["id"] = id
    codec = registry.get_codec(conf)
    return codec


def cfcodec_from_dataset(dataset: h5py.Dataset) -> Codec | None:
    attributes = {attr: dataset.attrs[attr] for attr in dataset.attrs}
    mapping = {}
    if "scale_factor" in attributes:
        try:
            scale_factor = attributes["scale_factor"][0]
        except IndexError:
            scale_factor = attributes["scale_factor"]
        mapping["scale_factor"] = float(1 / scale_factor)
    else:
        mapping["scale_factor"] = 1
    if "add_offset" in attributes:
        try:
            offset = attributes["add_offset"][0]
        except IndexError:
            offset = attributes["add_offset"]
        mapping["add_offset"] = float(offset)
    else:
        mapping["add_offset"] = 0
    if mapping["scale_factor"] != 1 or mapping["add_offset"] != 0:
        float_dtype = _choose_float_dtype(dtype=dataset.dtype, mapping=mapping)
        target_dtype = np.dtype(float_dtype)
        codec = FixedScaleOffset(
            offset=mapping["add_offset"],
            scale=mapping["scale_factor"],
            dtype=target_dtype,
            astype=dataset.dtype,
        )
        cfcodec = CFCodec(target_dtype=target_dtype, codec=codec)
        return cfcodec
    else:
        return None


def codecs_from_dataset(dataset: h5py.Dataset) -> List[Codec]:
    codecs = []
    for filter_id, filter_properties in dataset._filters.items():
        codec = _filter_to_codec(filter_id, filter_properties)
        codecs.append(codec)
    return codecs
