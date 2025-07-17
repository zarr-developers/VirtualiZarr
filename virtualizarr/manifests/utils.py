from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Union

import numpy as np
from zarr import Array
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    parse_dimension_names,
    parse_shapelike,
)
from zarr.dtype import parse_data_type

from virtualizarr.codecs import convert_to_codec_pipeline, get_codecs

if TYPE_CHECKING:
    from .array import ManifestArray


def construct_chunk_pattern(chunk_key_encoding: Literal[".", "/"]) -> str:
    """
    Produces a pattern for finding a chunk indices from key within a Zarr store using [re.match][] or [re.search][].

    Parameters
    ----------
    chunk_key_encoding
        The chunk key separator used in the Zarr store.

    Returns
    -------
    String representation of regular expression for a chunk key index
    """

    integer_pattern = r"([1-9]+\d*|0)"  # matches 0 or an unsigned integer that does not begin with zero
    separator = (
        rf"\{chunk_key_encoding}" if chunk_key_encoding == "." else chunk_key_encoding
    )
    pattern = rf"{integer_pattern}+({separator}{integer_pattern})*$"  # matches 1 integer, optionally followed by more integers each separated by a separator (i.e. a period)
    return pattern


def create_v3_array_metadata(
    shape: tuple[int, ...],
    data_type: np.dtype,
    chunk_shape: tuple[int, ...],
    chunk_key_encoding: ChunkKeyEncodingLike = {"name": "default"},
    fill_value: Any = None,
    codecs: Optional[list[Dict[str, Any]]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    dimension_names: Iterable[str] | None = None,
) -> ArrayV3Metadata:
    """
    Create an ArrayV3Metadata instance with standard configuration.
    This function encapsulates common patterns used across different parsers.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array
    data_type : np.dtype
        The numpy dtype of the array
    chunk_shape : tuple[int, ...]
        The shape of each chunk
    chunk_key_encoding : ChunkKeyEncodingLike
        The mapping from chunk grid cell coordinates to keys.
    fill_value : Any, optional
        The fill value for the array
    codecs : list[Dict[str, Any]], optional
        List of codec configurations
    attributes : Dict[str, Any], optional
        Additional attributes for the array
    dimension_names : tuple[str], optional
        Names of the dimensions

    Returns
    -------
    ArrayV3Metadata
        A configured ArrayV3Metadata instance with standard defaults
    """
    zdtype = parse_data_type(data_type, zarr_format=3)
    return ArrayV3Metadata(
        shape=shape,
        data_type=zdtype,
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_shape},
        },
        chunk_key_encoding=chunk_key_encoding,
        fill_value=zdtype.default_scalar() if fill_value is None else fill_value,
        codecs=convert_to_codec_pipeline(
            codecs=codecs or [],
            dtype=data_type,
        ),
        attributes=attributes or {},
        dimension_names=dimension_names,
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


def _remove_element_at_position(t: tuple[int, ...], pos: int) -> tuple[int, ...]:
    new_l = list(t)
    new_l.pop(pos)
    return tuple(new_l)


def check_no_partial_chunks_on_concat_axis(
    shapes: list[tuple[int, ...]], chunks: list[tuple[int, ...]], axis: int
):
    """Check that there are no partial chunks along the concatenation axis"""
    # loop over the arrays to be concatenated
    for i, (shape, chunk_shape) in enumerate(zip(shapes, chunks)):
        if shape[axis] % chunk_shape[axis] > 0:
            raise ValueError(
                "Cannot concatenate arrays with partial chunks because only regular chunk grids are currently supported. "
                f"Concat input {i} has array length {shape[axis]} along the concatenation axis which is not "
                f"evenly divisible by chunk length {chunk_shape[axis]}."
            )


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
    new_dimension_names: Iterable[str] | None | Literal["default"] = "default",
    new_attributes: dict | None = None,
) -> ArrayV3Metadata:
    """
    Update metadata to reflect a new shape and/or chunk shape.
    """
    # TODO this should really be upstreamed into zarr-python

    metadata_copy = old_metadata.to_dict().copy()

    if new_shape is not None:
        metadata_copy["shape"] = parse_shapelike(new_shape)  # type: ignore[assignment]
    if new_chunks is not None:
        metadata_copy["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(new_chunks)},
        }
    if new_dimension_names != "default":
        # need the option to use the literal string "default" as a sentinel value because None is a valid choice for zarr dimension_names
        metadata_copy["dimension_names"] = parse_dimension_names(new_dimension_names)
    if new_attributes is not None:
        metadata_copy["attributes"] = new_attributes

    # ArrayV3Metadata.from_dict removes extra keys zarr_format and node_type
    new_metadata = ArrayV3Metadata.from_dict(metadata_copy)
    return new_metadata
