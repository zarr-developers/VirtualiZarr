from typing import Any

import numpy as np
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.codec import Codec as ZarrCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.common import JSON
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,  # just the .zattrs (for one array or for the whole store/group)
)


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
    """
    Convert a decoded zarr array (.zarray) reference to an ArrayV3Metadata object.
    This function processes the given decoded Zarr array reference dictionary,
    to construct and return an ArrayV3Metadata object based on the provided information.

    Parameters:
    ----------
    decoded_arr_refs_zarray : dict
        A dictionary containing the decoded Zarr array reference information.
        Expected keys include "dtype", "fill_value", "zarr_format", "filters",
        "compressor", "chunks", and "shape".
    Returns:
    -------
    ArrayV3Metadata
    Raises:
    ------
    ValueError
        If the Zarr format specified in the input dictionary is not 2 or 3.
    """
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
