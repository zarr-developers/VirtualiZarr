from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from zarr import Array  # type: ignore
    from zarr.core.abc.codec import (  # type: ignore
        ArrayArrayCodec,
        ArrayBytesCodec,
        BytesBytesCodec,
    )

    from .manifests.array import ManifestArray

CodecPipeline = Tuple[
    Union["ArrayArrayCodec", "ArrayBytesCodec", "BytesBytesCodec"], ...
]


def get_codecs(array: Union["ManifestArray", "Array"]) -> CodecPipeline:
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
    ImportError
        If `zarr` is required but not installed.
    ValueError
        If the array type is unsupported or the array's metadata is not in zarr v3 format.
    NotImplementedError
        If zarr-python v3 is not installed.
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


def _get_manifestarray_codecs(array: "ManifestArray") -> CodecPipeline:
    """Get zarr v3 codec pipeline for a ManifestArray."""
    if array.metadata.zarr_format != 3:
        raise ValueError(
            "Only zarr v3 format is supported. Please convert your array metadata to v3 format."
        )
    return array.metadata.codecs


def _is_zarr_array(array: object) -> bool:
    """Check if the array is an instance of Zarr Array."""
    try:
        from zarr import Array

        return isinstance(array, Array)
    except ImportError:
        return False


def _get_zarr_array_codecs(array: "Array") -> CodecPipeline:
    """Get zarr v3 codec pipeline for a Zarr Array."""
    import zarr
    from packaging import version

    # Check that zarr-python v3 is installed
    required_version = "3.0.0b"
    installed_version = zarr.__version__
    if version.parse(installed_version) < version.parse(required_version):
        raise NotImplementedError(
            f"zarr-python v3 or higher is required, but version {installed_version} is installed."
        )

    from zarr.core.metadata import ArrayV3Metadata  # type: ignore[import-untyped]

    if not isinstance(array.metadata, ArrayV3Metadata):
        raise ValueError(
            "Only zarr v3 format arrays are supported. Please convert your array to v3 format."
        )

    return tuple(array.metadata.codecs)
