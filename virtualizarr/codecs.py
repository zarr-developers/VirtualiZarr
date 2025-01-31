from typing import TYPE_CHECKING, Union

from zarr.core.array import create_codec_pipeline

from virtualizarr.zarr import Codec

if TYPE_CHECKING:
    from zarr import Array  # type: ignore
    from zarr.core.abc.codec import (  # type: ignore
        ArrayArrayCodec,
        ArrayBytesCodec,
        BytesBytesCodec,
    )

    from .manifests.array import ManifestArray


def get_codecs(
    array: Union["ManifestArray", "Array"],
    normalize_to_zarr_v3: bool = False,
) -> Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]:
    """
    Get the codecs for either a ManifestArray or a Zarr Array.

    Parameters:
        array (Union[ManifestArray, ZarrArray]): The input array, either ManifestArray or Zarr Array.

    Returns:
        List[Optional[Codec]]: A list of codecs or an empty list if no codecs are found.

    Raises:
        ImportError: If `zarr` is required but not installed.
        ValueError: If the array type is unsupported.
    """
    if _is_manifest_array(array):
        return _get_manifestarray_codecs(array, normalize_to_zarr_v3)  # type: ignore[arg-type]

    if _is_zarr_array(array):
        return _get_zarr_array_codecs(array, normalize_to_zarr_v3)  # type: ignore[arg-type]

    raise ValueError("Unsupported array type or zarr is not installed.")


def _is_manifest_array(array: object) -> bool:
    """Check if the array is an instance of ManifestArray."""
    try:
        from .manifests.array import ManifestArray

        return isinstance(array, ManifestArray)
    except ImportError:
        return False


def _get_manifestarray_codecs(
    array: "ManifestArray",
    normalize_to_zarr_v3: bool = False,
) -> Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]:
    """Get codecs for a ManifestArray based on its zarr_format."""
    return create_codec_pipeline(array.zarray)


def _is_zarr_array(array: object) -> bool:
    """Check if the array is an instance of Zarr Array."""
    try:
        from zarr import Array

        return isinstance(array, Array)
    except ImportError:
        return False


def _get_zarr_array_codecs(
    array: "Array",
    # Q: Note sure if we need this anymore
    # normalize_to_zarr_v3: bool = False,
) -> Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]:
    """Get codecs for a Zarr Array based on its format."""
    return create_codec_pipeline(array)
