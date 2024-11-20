from typing import TYPE_CHECKING, Union

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
        return _get_manifestarray_codecs(array)  # type: ignore[arg-type]

    if _is_zarr_array(array):
        return _get_zarr_array_codecs(array)  # type: ignore[arg-type]

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
) -> Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]:
    """Get codecs for a ManifestArray based on its zarr_format."""
    if array.zarray.zarr_format == 3:
        # I think this returns a list from a tuple of Zarr Codec objects (zarr.abc.codec)
        return array.zarray._v3_codec_pipeline()
    elif array.zarray.zarr_format == 2:
        # This returns virtualizarr.zarr.ZArray.codec which is
        # its own data class with compression as a dict and filters as a list of dicts
        return array.zarray.codec
    else:
        raise ValueError("Unsupported zarr_format for ManifestArray.")


def _is_zarr_array(array: object) -> bool:
    """Check if the array is an instance of Zarr Array."""
    try:
        from zarr import Array

        return isinstance(array, Array)
    except ImportError:
        return False


def _get_zarr_array_codecs(
    array: "Array",
) -> Union[Codec, tuple["ArrayArrayCodec | ArrayBytesCodec | BytesBytesCodec", ...]]:
    """Get codecs for a Zarr Array based on its format."""
    import zarr
    from packaging import version

    # Check that zarr-python v3 is installed
    required_version = "3.0.0b"
    installed_version = zarr.__version__
    if version.parse(installed_version) < version.parse(required_version):
        raise NotImplementedError(
            f"zarr-python v3 or higher is required, but version {installed_version} is installed."
        )
    # For zarr format v3
    if hasattr(array, "metadata") and array.metadata.zarr_format == 3:
        return tuple(array.metadata.codecs)
    # For zarr format v2
    elif hasattr(array, "metadata") and array.metadata.zarr_format == 2:
        return Codec(
            compressor=array.metadata.compressor, filters=array.metadata.filters
        )  # type: ignore[attr-defined]
