from typing import List, Tuple, Union

import h5py
import hdf5plugin
import numcodecs.registry as registry
from numcodecs.abc import Codec
from pydantic import BaseModel, validator

_non_standard_filters = {
    "gzip": "zlib"
}


class BloscProperties(BaseModel):
    blocksize: int
    clevel: int
    shuffle: int
    cname: str

    @validator("cname", pre=True)
    def get_cname_from_code(cls, v):
        blosc_compressor_codes = {
            value: key for key, value in hdf5plugin._filters.Blosc._Blosc__COMPRESSIONS.items()
        }
        return blosc_compressor_codes[v]


def _filter_to_codec(filter_id: str, filter_properties: Union[int, Tuple] = None) -> Codec:
    try:
        id = int(filter_id)
    except ValueError:
        id = filter_id

    if isinstance(id, str):
        if id in _non_standard_filters.keys():
            id = _non_standard_filters[id]
        conf = {"id": id}
        if id == "zlib":
            conf["level"] = filter_properties
    elif isinstance(id, int):
        filter = hdf5plugin.get_filters(id)[0]
        id = filter.filter_name
        if id == "blosc":
            blosc_props = BloscProperties(**{k: v for k, v in
                                             zip(BloscProperties.__fields__.keys(),
                                                 filter_properties[-4:])})
            conf = blosc_props.model_dump()
            conf["id"] = id

    codec = registry.get_codec(conf)
    return codec


def codecs_from_dataset(dataset: h5py.Dataset) -> List[Codec]:
    codecs = []
    for filter_id, filter_properties in dataset._filters.items():
        codec = _filter_to_codec(filter_id, filter_properties)
        codecs.append(codec)
    return codecs
