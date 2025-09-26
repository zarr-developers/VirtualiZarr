from typing import TYPE_CHECKING, Any, Tuple, Union

import numpy as np
import zarr
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.abc.codec import Codec as ZarrCodec
from zarr.codecs import BytesCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from .manifests.array import ManifestArray

CodecPipeline = Tuple[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec, ...]

DeconstructedCodecPipeline = tuple[
    tuple[ArrayArrayCodec, ...],  # Array-to-array transformations
    ArrayBytesCodec | None,  # Array-to-bytes conversion
    tuple[BytesBytesCodec, ...],  # Bytes-to-bytes transformations
]


def zarr_codec_config_to_v3(num_codec: dict) -> dict:
    """
    Convert a numcodecs codec into a zarr v3 configurable.
    """
    # TODO: Special case Blosc codec
    if num_codec["id"].startswith("numcodecs."):
        return num_codec

    num_codec_copy = num_codec.copy()
    name = "numcodecs." + num_codec_copy.pop("id")
    return {"name": name, "configuration": num_codec_copy}


def zarr_codec_config_to_v2(num_codec: dict) -> dict:
    """
    Convert a numcodecs codec into a zarr v2 configurable.
    """
    # TODO: Special case Blosc codec
    if name := num_codec.get("name", None):
        return {"id": name, **num_codec["configuration"]}
    elif num_codec.get("id", None):
        return num_codec
    else:
        raise ValueError(f"Expected a valid Zarr V2 or V3 codec dict, got {num_codec}")


def extract_codecs(
    codecs: CodecPipeline,
) -> DeconstructedCodecPipeline:
    """Extracts various codec types."""

    arrayarray_codecs: tuple[ArrayArrayCodec, ...] = ()
    arraybytes_codec: ArrayBytesCodec | None = None
    bytesbytes_codecs: tuple[BytesBytesCodec, ...] = ()
    for codec in codecs:
        if not isinstance(codec, (ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec)):
            raise TypeError(
                "All codecs must be valid zarr v3 codecs, "
                f"but supplied codec {codec} does not subclass any of "
                "``zarr.abc.codec.ArrayArrayCodec``, ``zarr.abc.codec.ArrayBytesCodec``, or ``zarr.abc.codec.BytesBytesCodec``. "
                "Please see https://zarr.readthedocs.io/en/stable/user-guide/extending.html#custom-codecs for details on how to specify custom zarr codecs."
            )

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
    Convert list of codecs to valid BatchedCodecPipeline.

    Parameters
    ----------
    dtype
    codecs

    Returns
    -------
    BatchedCodecPipeline
    """
    from zarr.registry import get_codec_class

    zarr_codecs: tuple[ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec, ...] = ()
    if codecs and len(codecs) > 0:
        zarr_codecs = tuple(
            get_codec_class(codec["name"]).from_dict(codec) for codec in codecs
        )

    # It would be nice to use zarr.core.codec_pipeline.codecs_from_list here but that function requires
    # array array codecs and array bytes codecs to already be present in the list and in the correct order.
    arrayarray_codecs, arraybytes_codec, bytesbytes_codecs = extract_codecs(zarr_codecs)

    if arraybytes_codec is None:
        if dtype.byteorder == ">":
            arraybytes_codec = BytesCodec(endian="big")
        else:
            arraybytes_codec = BytesCodec()

    codec_pipeline = BatchedCodecPipeline(
        array_array_codecs=arrayarray_codecs,
        array_bytes_codec=arraybytes_codec,
        bytes_bytes_codecs=bytesbytes_codecs,
        batch_size=1,
    )

    return codec_pipeline


def get_codec_config(codec: ZarrCodec) -> dict[str, Any]:
    """
    Extract configuration from a codec, handling both zarr-python and numcodecs codecs.
    """

    if hasattr(codec, "codec_config"):
        return codec.codec_config
    elif hasattr(codec, "get_config"):
        return codec.get_config()
    elif hasattr(codec, "_zstd_codec"):
        # related issue: https://github.com/zarr-developers/VirtualiZarr/issues/514
        # very silly workaround. codec.to_dict for zstd gives:
        # {'name': 'zstd', 'configuration': {'level': 0, 'checksum': False}}
        # which when passed through ArrayV2Metadata -> numcodecs.get_codec gives the error:
        # *** numcodecs.errors.UnknownCodecError: codec not available: 'None'
        # if codec._zstd_codec.get_config() : {'id': 'zstd', 'level': 0, 'checksum': False}
        # is passed to numcodecs.get_codec. It works fine.
        return codec._zstd_codec.get_config()
    elif hasattr(codec, "to_dict"):
        return codec.to_dict()
    else:
        raise ValueError(f"Unable to parse codec configuration: {codec}")


def get_codecs(array: Union["ManifestArray", "zarr.Array"]) -> CodecPipeline:
    """
    Get the zarr v3 codec pipeline for either a ManifestArray or a Zarr Array.

    Parameters
    ----------
    array : Union[ManifestArray, Array]
        The input array, either ManifestArray or Zarr Array.

    Returns
    -------
    CodecPipeline
        A tuple of zarr v3 codecs representing the codec pipeline.

    Raises
    ------
    ValueError
        If the array type is unsupported or the array's metadata is not in zarr v3 format.
    """
    if not isinstance(array.metadata, ArrayV3Metadata):
        raise ValueError(
            "Only zarr v3 format arrays are supported. Please convert your array to v3 format."
        )

    return array.metadata.codecs
