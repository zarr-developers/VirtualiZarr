import dataclasses
from typing import TYPE_CHECKING, Any, Literal, NewType

import numpy as np
from zarr.abc.codec import Codec as ZarrCodec
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    pass

# TODO replace these with classes imported directly from Zarr? (i.e. Zarr Object Models)
ZAttrs = NewType(
    "ZAttrs", dict[str, Any]
)  # just the .zattrs (for one array or for the whole store/group)
FillValueT = bool | str | float | int | list | None
ZARR_FORMAT = Literal[2, 3]

ZARR_DEFAULT_FILL_VALUE: dict[str, FillValueT] = {
    # numpy dtypes's hierarchy lets us avoid checking for all the widths
    # https://numpy.org/doc/stable/reference/arrays.scalars.html
    np.dtype("bool").kind: False,
    np.dtype("int").kind: 0,
    np.dtype("float").kind: 0.0,
    np.dtype("complex").kind: [0.0, 0.0],
    np.dtype("datetime64").kind: 0,
}
"""
The value and format of the fill_value depend on the `data_type` of the array.
See here for spec:
https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value
"""


@dataclasses.dataclass
class Codec:
    compressors: list[dict] | None = None
    filters: list[dict] | None = None


@dataclasses.dataclass
class ZArray:
    """Just the .zarray information"""

    # TODO will this work for V3?

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    fill_value: FillValueT = dataclasses.field(default=None)
    order: Literal["C", "F"] = "C"
    compressor: dict | None = None
    filters: list[dict] | None = None
    zarr_format: Literal[2, 3] = 2

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.chunks):
            raise ValueError(
                "Dimension mismatch between array shape and chunk shape. "
                f"Array shape {self.shape} has ndim={self.shape} but chunk shape {self.chunks} has ndim={len(self.chunks)}"
            )

        if isinstance(self.dtype, str):
            # Convert dtype string to numpy.dtype
            self.dtype = np.dtype(self.dtype)

        if self.fill_value is None:
            self.fill_value = ZARR_DEFAULT_FILL_VALUE.get(self.dtype.kind, 0.0)

    @property
    def codec(self) -> Codec:
        """For comparison against other arrays."""
        return Codec(compressor=self.compressor, filters=self.filters)

    @classmethod
    def from_kerchunk_refs(cls, decoded_arr_refs_zarray) -> "ArrayV3Metadata":
        # coerce type of fill_value as kerchunk can be inconsistent with this
        dtype = np.dtype(decoded_arr_refs_zarray["dtype"])
        fill_value = decoded_arr_refs_zarray["fill_value"]
        if np.issubdtype(dtype, np.floating) and (
            fill_value is None or fill_value == "NaN" or fill_value == "nan"
        ):
            fill_value = np.nan

        zarr_format = int(decoded_arr_refs_zarray["zarr_format"])
        if zarr_format not in (2, 3):
            raise ValueError(f"Zarr format must be 2 or 3, but got {zarr_format}")
        filters = (
            decoded_arr_refs_zarray.get("filters", []) or []
        )  # Ensure filters is a list
        compressor = decoded_arr_refs_zarray.get("compressor")  # Might be None

        # Ensure compressor is a list before unpacking
        codec_configs = [*filters, *(compressor if compressor is not None else [])]
        numcodec_configs = [
            _num_codec_config_to_configurable(config) for config in codec_configs
        ]
        return ArrayV3Metadata(
            chunk_grid={
                "name": "regular",
                "configuration": {
                    "chunk_shape": tuple(decoded_arr_refs_zarray["chunks"])
                },
            },
            codecs=convert_to_codec_pipeline(
                dtype=dtype,
                codecs=numcodec_configs,
            ),
            data_type=dtype,
            fill_value=fill_value,
            shape=tuple(decoded_arr_refs_zarray["shape"]),
            chunk_key_encoding={"name": "default"},
            attributes={},
            dimension_names=None,
        )

    def dict(self) -> dict[str, Any]:
        zarray_dict = dataclasses.asdict(self)
        return zarray_dict

    def to_kerchunk_json(self) -> str:
        import ujson

        zarray_dict = self.dict()
        if zarray_dict["fill_value"] is np.nan:
            zarray_dict["fill_value"] = None
        return ujson.dumps(zarray_dict)

    # ZArray.dict seems to shadow "dict", so we need the type ignore in
    # the signature below.
    def replace(
        self,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | str | None = None,
        fill_value: FillValueT = None,
        order: Literal["C", "F"] | None = None,
        compressor: "dict | None" = None,  # type: ignore[valid-type]
        filters: list[dict] | None = None,  # type: ignore[valid-type]
        zarr_format: Literal[2, 3] | None = None,
    ) -> "ZArray":
        """
        Convenience method to create a new ZArray from an existing one by altering only certain attributes.
        """
        replacements: dict[str, Any] = {}
        if shape is not None:
            replacements["shape"] = shape
        if chunks is not None:
            replacements["chunks"] = chunks
        if dtype is not None:
            replacements["dtype"] = dtype
        if fill_value is not None:
            replacements["fill_value"] = fill_value
        if order is not None:
            replacements["order"] = order
        if compressor is not None:
            replacements["compressor"] = compressor
        if filters is not None:
            replacements["filters"] = filters
        if zarr_format is not None:
            replacements["zarr_format"] = zarr_format
        return dataclasses.replace(self, **replacements)

    def _v3_codec_pipeline(self) -> tuple["ZarrCodec", ...]:
        """
        Convert the compressor, filters, and dtype to a pipeline of ZarrCodecs.
        """
        filters = self.filters or []
        return convert_to_codec_pipeline(
            codecs=[*filters, self.compressor], dtype=self.dtype
        )

    def serializer(self) -> Any:
        """
        testing
        """
        try:
            from zarr.core.metadata.v3 import (  # type: ignore[import-untyped]
                parse_codecs,
            )
        except ImportError:
            raise ImportError("zarr v3 is required to generate v3 codec pipelines")
        # https://github.com/zarr-developers/zarr-python/pull/1944#issuecomment-2151994097
        # "If no ArrayBytesCodec is supplied, we can auto-add a BytesCodec"
        bytes = dict(
            name="bytes", configuration={}
        )  # TODO need to handle endianess configuration
        return parse_codecs([bytes])[0]


def ceildiv(a: int, b: int) -> int:
    """
    Ceiling division operator for integers.

    See https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(a // -b)


def determine_chunk_grid_shape(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(ceildiv(length, chunksize) for length, chunksize in zip(shape, chunks))


def _num_codec_config_to_configurable(num_codec: dict) -> dict:
    """
    Convert a numcodecs codec into a zarr v3 configurable.
    """
    if num_codec["id"].startswith("numcodecs."):
        return num_codec

    num_codec_copy = num_codec.copy()
    name = "numcodecs." + num_codec_copy.pop("id")
    # name = num_codec_copy.pop("id")
    return {"name": name, "configuration": num_codec_copy}


def extract_codecs(
    codecs: list[ZarrCodec] | None = [],
) -> tuple[list[dict], list[dict]]:
    """Extracts filters and compressor from Zarr v3 metadata."""
    from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec

    arrayarray_codecs, bytesbytes_codecs = [], []
    for codec in codecs:
        if isinstance(codec, ArrayArrayCodec):
            arrayarray_codecs.append(codec)
        if isinstance(codec, BytesBytesCodec):
            bytesbytes_codecs.append(codec)
    return arrayarray_codecs, bytesbytes_codecs


def convert_to_codec_pipeline(
    dtype: np.dtype,
    codecs: list[dict] | None = [],
) -> tuple[ZarrCodec, ...]:
    """
    Convert compressor, filters, serializer, and dtype to a pipeline of ZarrCodecs.

    Parameters
    ----------
    dtype : Any
        The data type.
    codecs: list[ZarrCodec] | None

    Returns
    -------
    Tuple[ZarrCodec, ...]
        A tuple of ZarrCodecs.
    """
    from zarr.core.array import _get_default_chunk_encoding_v3
    from zarr.registry import get_codec_class

    if codecs and len(codecs) > 0:
        zarr_codecs = [
            get_codec_class(codec["name"]).from_dict(codec) for codec in codecs
        ]
        arrayarray_codecs, bytesbytes_codecs = extract_codecs(zarr_codecs)
    else:
        arrayarray_codecs, bytesbytes_codecs = [], []
    # FIXME: using private zarr-python function
    # we can also use zarr_config.get("array.v3_default_serializer").get("numeric"), but requires rewriting a lot of this function
    arraybytes_codecs = _get_default_chunk_encoding_v3(dtype)[1]
    codec_pipeline = (*arrayarray_codecs, arraybytes_codecs, *bytesbytes_codecs)

    return codec_pipeline


def convert_v3_to_v2_metadata(
    v3_metadata: ArrayV3Metadata, fill_value: Any = None
) -> ArrayV2Metadata:
    """
    Convert ArrayV3Metadata to ArrayV2Metadata.

    Parameters
    ----------
    v3_metadata : ArrayV3Metadata
        The metadata object in v3 format.

    Returns
    -------
    ArrayV2Metadata
        The metadata object in v2 format.
    """
    import warnings

    filters, compressors = extract_codecs(v3_metadata.codecs)
    if compressors:
        compressor = compressors[0]
        compressor_config = compressor.codec_config
        if len(compressors) > 1:
            warnings.warn(
                "Multiple compressors detected. Only the first one will be used."
            )
    else:
        compressor_config = None

    filter_configs = [filter.codec_config for filter in filters]

    v2_metadata = ArrayV2Metadata(
        shape=v3_metadata.shape,
        dtype=v3_metadata.data_type.to_numpy(),
        chunks=v3_metadata.chunk_grid.chunk_shape,
        fill_value=fill_value or v3_metadata.fill_value,
        compressor=compressor_config,
        filters=filter_configs,
        order="C",
        attributes=v3_metadata.attributes,
        dimension_separator=".",  # Assuming '.' as default dimension separator
    )
    return v2_metadata


def to_kerchunk_json(v2_metadata: ArrayV2Metadata) -> str:
    import ujson

    zarray_dict = v2_metadata.to_dict()
    if zarray_dict["filters"]:
        zarray_dict["filters"] = [
            codec.get_config() for codec in zarray_dict["filters"]
        ]
    if zarray_dict["compressor"]:
        zarray_dict["compressor"] = zarray_dict["compressor"].get_config()

    # TODO: Use NumpyEncoder here?
    fill_value = zarray_dict["fill_value"]
    if fill_value is not None and np.isnan(fill_value):
        zarray_dict["fill_value"] = None
    if isinstance(fill_value, np.generic):
        zarray_dict["fill_value"] = fill_value.item()
    return ujson.dumps(zarray_dict)
