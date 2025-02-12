from typing import TYPE_CHECKING, Tuple, Union

import zarr
from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec
from zarr.core.metadata.v3 import ArrayV3Metadata

if TYPE_CHECKING:
    from .manifests.array import ManifestArray

CodecPipeline = Tuple[
    Union["ArrayArrayCodec", "ArrayBytesCodec", "BytesBytesCodec"], ...
]


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
