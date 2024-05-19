from typing import List, Optional, Tuple, TypedDict, Union

import h5py
import hdf5plugin
import numcodecs.registry as registry
from numcodecs.abc import Codec
from pydantic import BaseModel, validator

_non_standard_filters = {"gzip": "zlib"}


class BloscProperties(BaseModel):
    blocksize: int
    clevel: int
    shuffle: int
    cname: str

    @validator("cname", pre=True)
    def get_cname_from_code(cls, v):
        blosc_compressor_codes = {
            value: key
            for key, value in hdf5plugin._filters.Blosc._Blosc__COMPRESSIONS.items()
        }
        return blosc_compressor_codes[v]


def _filter_to_codec(
    filter_id: str, filter_properties: Union[int, None, Tuple] = None
) -> Codec:
    id_int = None
    id_str = None
    try:
        id_int = int(filter_id)
    except ValueError:
        id_str = filter_id

    if id_str:
        if id_str in _non_standard_filters.keys():
            id = _non_standard_filters[id_str]
        else:
            id = id_str
        conf = {"id": id}
        if id == "zlib":
            conf["level"] = filter_properties  # type: ignore[assignment]
    if id_int:
        filter = hdf5plugin.get_filters(id_int)[0]
        id = filter.filter_name
        if id == "blosc" and isinstance(filter_properties, tuple):
            blosc_props = BloscProperties(
                **{
                    k: v
                    for k, v in zip(
                        BloscProperties.__fields__.keys(), filter_properties[-4:]
                    )
                }
            )
            conf = blosc_props.model_dump()  # type: ignore[assignment]
            conf["id"] = id

    codec = registry.get_codec(conf)
    return codec


def codecs_from_dataset(dataset: h5py.Dataset) -> List[Codec]:
    codecs = []
    for filter_id, filter_properties in dataset._filters.items():
        codec = _filter_to_codec(filter_id, filter_properties)
        codecs.append(codec)
    return codecs
