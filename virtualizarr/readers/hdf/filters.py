import dataclasses
from typing import TYPE_CHECKING, List, Tuple, TypedDict, Union

import numcodecs.registry as registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs.fixedscaleoffset import FixedScaleOffset
from xarray.coding.variables import _choose_float_dtype

from virtualizarr.utils import soft_import

if TYPE_CHECKING:
    import h5py  # type: ignore
    from h5py import Dataset, Group  # type: ignore

h5py = soft_import("h5py", "For reading hdf files", strict=False)
if h5py:
    Dataset = h5py.Dataset
    Group = h5py.Group
else:
    Dataset = dict()
    Group = dict()

hdf5plugin = soft_import(
    "hdf5plugin", "For reading hdf files with filters", strict=False
)
imagecodecs = soft_import(
    "imagecodecs", "For reading hdf files with filters", strict=False
)


_non_standard_filters = {
    "gzip": "zlib",
    "lzf": "imagecodecs_lzf",
}

_hdf5plugin_imagecodecs = {"lz4": "imagecodecs_lz4h5", "bzip2": "imagecodecs_bz2"}


@dataclasses.dataclass
class BloscProperties:
    blocksize: int
    clevel: int
    shuffle: int
    cname: str

    def __post_init__(self):
        blosc_compressor_codes = {
            value: key
            for key, value in hdf5plugin._filters.Blosc._Blosc__COMPRESSIONS.items()
        }
        self.cname = blosc_compressor_codes[self.cname]


@dataclasses.dataclass
class ZstdProperties:
    level: int


@dataclasses.dataclass
class ShuffleProperties:
    elementsize: int


@dataclasses.dataclass
class ZlibProperties:
    level: int


class CFCodec(TypedDict):
    target_dtype: np.dtype
    codec: Codec


def _filter_to_codec(
    filter_id: str, filter_properties: Union[int, None, Tuple] = None
) -> Codec:
    """
    Convert an h5py filter to an equivalent numcodec

    Parameters
    ----------
    filter_id: str
        An h5py filter id code.
    filter_properties : int or None or Tuple
        A single or Tuple of h5py filter configuration codes.

    Returns
    -------
        A numcodec codec
    """
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
            zlib_props = ZlibProperties(level=filter_properties)  # type: ignore
            conf = dataclasses.asdict(zlib_props)
        if id == "shuffle" and isinstance(filter_properties, tuple):
            shuffle_props = ShuffleProperties(elementsize=filter_properties[0])
            conf = dataclasses.asdict(shuffle_props)
        conf["id"] = id  # type: ignore[assignment]
    if id_int:
        filter = hdf5plugin.get_filters(id_int)[0]
        id = filter.filter_name
        if id in _hdf5plugin_imagecodecs.keys():
            id = _hdf5plugin_imagecodecs[id]
        if id == "blosc" and isinstance(filter_properties, tuple):
            blosc_fields = [field.name for field in dataclasses.fields(BloscProperties)]
            blosc_props = BloscProperties(
                **{k: v for k, v in zip(blosc_fields, filter_properties[-4:])}
            )
            conf = dataclasses.asdict(blosc_props)
        if id == "zstd" and isinstance(filter_properties, tuple):
            zstd_props = ZstdProperties(level=filter_properties[0])
            conf = dataclasses.asdict(zstd_props)
        conf["id"] = id
    codec = registry.get_codec(conf)
    return codec


def cfcodec_from_dataset(dataset: Dataset) -> Codec | None:
    """
    Converts select h5py dataset CF convention attrs to CFCodec

    Parameters
    ----------
    dataset: h5py.Dataset
       An h5py dataset.

    Returns
    -------
    CFCodec
        A CFCodec.
    """
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


def codecs_from_dataset(dataset: Dataset) -> List[Codec]:
    """
    Extracts a list of numcodecs from an h5py dataset

    Parameters
    ----------
    dataset: h5py.Dataset
       An h5py dataset.

    Returns
    -------
    list
        A list of numcodecs codecs.
    """
    codecs = []
    for filter_id, filter_properties in dataset._filters.items():
        codec = _filter_to_codec(filter_id, filter_properties)
        codecs.append(codec)
    return codecs
