from typing import List, Tuple, Union

import h5py
import numcodecs.registry as registry
from numcodecs.abc import Codec

_non_standard_filters = {
    "gzip": "zlib"
}


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

    codec = registry.get_codec(conf)
    return codec


def codecs_from_dataset(dataset: h5py.Dataset) -> List[Codec]:
    codecs = []
    for filter_id, filter_properties in dataset._filters.items():
        codec = _filter_to_codec(filter_id, filter_properties)
        codecs.append(codec)
    return codecs
