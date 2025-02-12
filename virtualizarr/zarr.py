import dataclasses
from typing import TYPE_CHECKING, Any, Literal, NewType

import numpy as np
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.codec import Codec as ZarrCodec
from zarr.core.common import JSON
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    try:
        from zarr.abc.codec import Codec as ZarrCodec
    except ImportError:
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


from virtualizarr.codecs import CodecPipeline


def extract_codecs(
    codecs: CodecPipeline,
) -> tuple[
    tuple[ArrayArrayCodec, ...], ArrayBytesCodec | None, tuple[BytesBytesCodec, ...]
]:
    """Extracts various codec types."""
    arrayarray_codecs: tuple[ArrayArrayCodec, ...] = ()
    arraybytes_codec: ArrayBytesCodec | None = None
    bytesbytes_codecs: tuple[BytesBytesCodec, ...] = ()
    for codec in codecs:
        if isinstance(codec, ArrayArrayCodec):
            arrayarray_codecs += (codec,)
        if isinstance(codec, ArrayBytesCodec):
            arraybytes_codec = codec
        if isinstance(codec, BytesBytesCodec):
            bytesbytes_codecs += (codec,)
    return (arrayarray_codecs, arraybytes_codec, bytesbytes_codecs)


from zarr.core.codec_pipeline import BatchedCodecPipeline


def convert_to_codec_pipeline(
    dtype: np.dtype,
    codecs: list[dict] | None = [],
) -> BatchedCodecPipeline:
    """
    Convert compressor, filters, serializer, and dtype to a pipeline of ZarrCodecs.

    Parameters
    ----------
    dtype : Any
        The data type.
    codecs: list[ZarrCodec] | None

    Returns
    -------
    BatchedCodecPipeline
    """
    from zarr.core.array import _get_default_chunk_encoding_v3
    from zarr.registry import get_codec_class

    zarr_codecs: tuple[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec, ...] = ()
    if codecs and len(codecs) > 0:
        zarr_codecs = tuple(
            get_codec_class(codec["name"]).from_dict(codec) for codec in codecs
        )

    # (aimee): I would like to use zarr.core.codec_pipeline.codecs_from_list here but it requires array array codecs before array bytes codecs,
    # which I don't think is actually a requirement.
    arrayarray_codecs, arraybytes_codec, bytesbytes_codecs = extract_codecs(zarr_codecs)

    if arraybytes_codec is None:
        arraybytes_codec = _get_default_chunk_encoding_v3(dtype)[1]

    codec_pipeline = BatchedCodecPipeline(
        array_array_codecs=arrayarray_codecs,
        array_bytes_codec=arraybytes_codec,
        bytes_bytes_codecs=bytesbytes_codecs,
        batch_size=1,
    )

    return codec_pipeline


def _get_codec_config(codec: ZarrCodec) -> dict[str, JSON]:
    """
    Extract configuration from a codec, handling both zarr-python and numcodecs codecs.
    """
    if hasattr(codec, "codec_config"):
        return codec.codec_config
    elif hasattr(codec, "get_config"):
        return codec.get_config()
    elif hasattr(codec, "codec_name"):
        # If we can't get config, try to get the name and configuration directly
        # This assumes the codec follows the v3 spec format
        return {
            "id": codec.codec_name.replace("numcodecs.", ""),
            **getattr(codec, "configuration", {}),
        }
    else:
        raise ValueError(f"Unable to parse codec configuration: {codec}")


def convert_v3_to_v2_metadata(
    v3_metadata: ArrayV3Metadata, fill_value: Any = None
) -> ArrayV2Metadata:
    """
    Convert ArrayV3Metadata to ArrayV2Metadata.

    Parameters
    ----------
    v3_metadata : ArrayV3Metadata
        The metadata object in v3 format.
    fill_value : Any, optional
        Override the fill value from v3 metadata.

    Returns
    -------
    ArrayV2Metadata
        The metadata object in v2 format.

    Notes
    -----
    The conversion handles the following cases:
    - Extracts compressor and filter configurations from v3 codecs
    - Preserves codec configurations regardless of codec implementation
    - Maintains backward compatibility with both zarr-python and numcodecs
    """
    import warnings

    array_filters: tuple[ArrayArrayCodec, ...]
    bytes_compressors: tuple[BytesBytesCodec, ...]
    array_filters, _, bytes_compressors = extract_codecs(v3_metadata.codecs)

    # Handle compressor configuration
    compressor_config: dict[str, JSON] | None = None
    if bytes_compressors:
        if len(bytes_compressors) > 1:
            warnings.warn(
                "Multiple compressors found in v3 metadata. Using the first compressor, "
                "others will be ignored. This may affect data compatibility.",
                UserWarning,
            )
        compressor_config = _get_codec_config(bytes_compressors[0])

    # Handle filter configurations
    filter_configs = [_get_codec_config(filter_) for filter_ in array_filters]
    v2_metadata = ArrayV2Metadata(
        shape=v3_metadata.shape,
        dtype=v3_metadata.data_type.to_numpy(),
        chunks=v3_metadata.chunks,
        fill_value=fill_value or v3_metadata.fill_value,
        compressor=compressor_config,
        filters=filter_configs,
        order="C",
        attributes=v3_metadata.attributes,
        dimension_separator=".",  # Assuming '.' as default dimension separator
    )
    return v2_metadata


def to_kerchunk_json(v2_metadata: ArrayV2Metadata) -> str:
    import json

    from virtualizarr.writers.kerchunk import NumpyEncoder

    zarray_dict: dict[str, JSON] = v2_metadata.to_dict()
    if v2_metadata.filters:
        zarray_dict["filters"] = [
            # we could also cast to json, but get_config is intended for serialization
            codec.get_config()
            for codec in v2_metadata.filters
            if codec is not None
        ]  # type: ignore[assignment]
    if v2_metadata.compressor:
        zarray_dict["compressor"] = v2_metadata.compressor.get_config()  # type: ignore[assignment]

    return json.dumps(zarray_dict, separators=(",", ":"), cls=NumpyEncoder)


def from_kerchunk_refs(decoded_arr_refs_zarray) -> "ArrayV3Metadata":
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
            "configuration": {"chunk_shape": tuple(decoded_arr_refs_zarray["chunks"])},
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
