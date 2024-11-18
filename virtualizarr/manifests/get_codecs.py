from typing import TYPE_CHECKING, Any, List, Optional, Union

from virtualizarr.zarr import Codec

if TYPE_CHECKING:
    from zarr import Array  # type: ignore

    from .array import ManifestArray


def get_codecs(array: Union["ManifestArray", "Array"]) -> Any:
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
        return _get_manifestarray_codecs(array)  # type: ignore[arg-type]

    if _is_zarr_array(array):
        return _get_zarr_array_codecs(array)  # type: ignore[arg-type]

    raise ValueError("Unsupported array type or zarr is not installed.")


def _is_manifest_array(array: object) -> bool:
    """Check if the array is an instance of ManifestArray."""
    try:
        from .array import ManifestArray

        return isinstance(array, ManifestArray)
    except ImportError:
        return False


def _get_manifestarray_codecs(array: "ManifestArray") -> List[Optional["Codec"]]:
    """Get codecs for a ManifestArray based on its zarr_format."""
    if array.zarray.zarr_format == 3:
        return list(array.zarray._v3_codec_pipeline())
    elif array.zarray.zarr_format == 2:
        return [array.zarray.codec]
    else:
        raise ValueError("Unsupported zarr_format for ManifestArray.")


def _is_zarr_array(array: object) -> bool:
    """Check if the array is an instance of Zarr Array."""
    try:
        from zarr import Array

        return isinstance(array, Array)
    except ImportError:
        return False


def _get_zarr_array_codecs(array: "Array") -> Any:
    """Get codecs for a Zarr Array based on its format."""
    try:
        # For Zarr v3
        if hasattr(array, "metadata") and hasattr(array.metadata, "codecs"):
            return array.metadata.codecs
        # For Zarr v2
        elif hasattr(array, "compressor") and hasattr(array, "filters"):
            return [Codec(compressor=array.compressor, filters=array.filters)]
        else:
            raise ValueError("Unsupported zarr_format for Zarr Array.")
    except ImportError:
        raise ImportError("zarr is not installed, but a Zarr Array was provided.")
