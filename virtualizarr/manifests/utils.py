from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Union

import numpy as np
from zarr import Array
from zarr.core.metadata.v3 import ArrayV3Metadata

from virtualizarr.codecs import convert_to_codec_pipeline, get_codecs

if TYPE_CHECKING:
    from .array import ManifestArray


def create_v3_array_metadata(
    shape: tuple[int, ...],
    data_type: np.dtype,
    chunk_shape: tuple[int, ...],
    fill_value: Any = None,
    codecs: Optional[list[Dict[str, Any]]] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> ArrayV3Metadata:
    """
    Create an ArrayV3Metadata instance with standard configuration.
    This function encapsulates common patterns used across different readers.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array
    data_type : np.dtype
        The numpy dtype of the array
    chunk_shape : tuple[int, ...]
        The shape of each chunk
    fill_value : Any, optional
        The fill value for the array
    codecs : list[Dict[str, Any]], optional
        List of codec configurations
    attributes : Dict[str, Any], optional
        Additional attributes for the array

    Returns
    -------
    ArrayV3Metadata
        A configured ArrayV3Metadata instance with standard defaults
    """
    return ArrayV3Metadata(
        shape=shape,
        data_type=data_type,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        chunk_key_encoding={"name": "default"},
        fill_value=fill_value,
        codecs=convert_to_codec_pipeline(
            codecs=codecs or [],
            dtype=data_type,
        ),
        attributes=attributes or {},
        dimension_names=None,
        storage_transformers=None,
    )


def check_same_dtypes(dtypes: list[np.dtype]) -> None:
    """Check all the dtypes are the same"""

    first_dtype, *other_dtypes = dtypes
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent dtypes: {other_dtype} vs {first_dtype}"
            )


def check_compatible_encodings(encoding1, encoding2):
    for key, value in encoding1.items():
        if key in encoding2:
            if encoding2[key] != value:
                raise ValueError(
                    f"Cannot concatenate arrays with different values for encoding key {key}: {encoding2[key]} != {value}"
                )


def check_same_codecs(codecs: list[Any]) -> None:
    first_codec, *other_codecs = codecs
    for codec in other_codecs:
        if codec != first_codec:
            raise NotImplementedError(
                "The ManifestArray class cannot concatenate arrays which were stored using different codecs, "
                f"But found codecs {first_codec} vs {codec} ."
                "See https://github.com/zarr-developers/zarr-specs/issues/288"
            )


def check_same_chunk_shapes(chunks_list: list[tuple[int, ...]]) -> None:
    """Check all the chunk shapes are the same"""

    first_chunks, *other_chunks_list = chunks_list
    for other_chunks in other_chunks_list:
        if other_chunks != first_chunks:
            raise ValueError(
                f"Cannot concatenate arrays with inconsistent chunk shapes: {other_chunks} vs {first_chunks} ."
                "Requires ZEP003 (Variable-length Chunks)."
            )


def check_same_ndims(ndims: list[int]) -> None:
    first_ndim, *other_ndims = ndims
    for other_ndim in other_ndims:
        if other_ndim != first_ndim:
            raise ValueError(
                f"Cannot concatenate arrays with differing number of dimensions: {first_ndim} vs {other_ndim}"
            )


def check_same_shapes(shapes: list[tuple[int, ...]]) -> None:
    first_shape, *other_shapes = shapes
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with differing shapes: {first_shape} vs {other_shape}"
            )


# TODO remove this once https://github.com/zarr-developers/zarr-python/issues/2929 is solved upstream
def metadata_identical(metadata1: ArrayV3Metadata, metadata2: ArrayV3Metadata) -> bool:
    """Checks the metadata of two zarr arrays are identical, including special treatment for NaN fill_values."""
    metadata_dict1 = metadata1.to_dict()
    metadata_dict2 = metadata2.to_dict()

    # fill_value is a special case because numpy NaNs cannot be compared using __eq__, see https://stackoverflow.com/a/10059796
    fill_value1 = metadata_dict1.pop("fill_value")
    fill_value2 = metadata_dict2.pop("fill_value")
    if np.isnan(fill_value1) and np.isnan(fill_value2):  # type: ignore[arg-type]
        fill_values_equal = fill_value1.dtype == fill_value2.dtype  # type: ignore[union-attr]
    else:
        fill_values_equal = fill_value1 == fill_value2

    # everything else in ArrayV3Metadata is a string, Enum, or Dataclass
    return fill_values_equal and metadata_dict1 == metadata_dict2


def _remove_element_at_position(t: tuple[int, ...], pos: int) -> tuple[int, ...]:
    new_l = list(t)
    new_l.pop(pos)
    return tuple(new_l)


def check_same_shapes_except_on_concat_axis(shapes: list[tuple[int, ...]], axis: int):
    """Check that shapes are compatible for concatenation"""

    shapes_without_concat_axis = [
        _remove_element_at_position(shape, axis) for shape in shapes
    ]

    first_shape, *other_shapes = shapes_without_concat_axis
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with shapes {[shape for shape in shapes]}"
            )


def check_combinable_zarr_arrays(
    arrays: Iterable[Union["ManifestArray", "Array"]],
) -> None:
    """
    The downside of the ManifestArray approach compared to the VirtualZarrArray concatenation proposal is that
    the result must also be a single valid zarr array, implying that the inputs must have the same dtype, codec etc.
    """
    check_same_dtypes([arr.dtype for arr in arrays])

    # Can't combine different codecs in one manifest
    # see https://github.com/zarr-developers/zarr-specs/issues/288
    check_same_codecs([get_codecs(arr) for arr in arrays])

    # Would require variable-length chunks ZEP
    check_same_chunk_shapes([arr.chunks for arr in arrays])


def check_compatible_arrays(
    ma: "ManifestArray", existing_array: "Array", append_axis: int
):
    check_combinable_zarr_arrays([ma, existing_array])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def copy_and_replace_metadata(
    old_metadata: ArrayV3Metadata,
    new_shape: list[int] | None = None,
    new_chunks: list[int] | None = None,
) -> ArrayV3Metadata:
    """
    Update metadata to reflect a new shape and/or chunk shape.
    """
    metadata_copy = old_metadata.to_dict().copy()
    metadata_copy["shape"] = new_shape  # type: ignore[assignment]
    if new_chunks is not None:
        metadata_copy["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(new_chunks)},
        }
    # ArrayV3Metadata.from_dict removes extra keys zarr_format and node_type
    new_metadata = ArrayV3Metadata.from_dict(metadata_copy)
    return new_metadata
