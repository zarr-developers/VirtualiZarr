import copy
import functools
import re
import typing
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Union, cast

import numpy as np
from zarr import Array
from zarr.core.chunk_key_encodings import ChunkKeyEncodingLike
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import (
    ArrayV3Metadata,
    parse_dimension_names,
    parse_shapelike,
)
from zarr.dtype import parse_data_type

from virtualizarr.codecs import convert_to_codec_pipeline, get_codecs

if TYPE_CHECKING:
    from zarr.core.metadata.v3 import RegularChunkGridMetadata

    from .array import ManifestArray
else:
    try:
        from zarr.core.metadata.v3 import RegularChunkGridMetadata  # zarr-python>3.1.6
    except ImportError:
        from zarr.core.metadata.v3 import (
            RegularChunkGrid as RegularChunkGridMetadata,  # zarr-python<=3.1.6
        )

ChunkKeySeparator = Literal[".", "/"]
# Tuple of literal values specified in the Literal type above
CHUNK_KEY_SEPARATORS = typing.get_args(ChunkKeySeparator)


def parse_manifest_index(
    key: str, separator: ChunkKeySeparator = ".", expand_pattern: bool = False
) -> tuple[int, ...]:
    """
    Extract the chunk index from a `key` (a.k.a `node`) that represents a chunk of
    data in a Zarr hierarchy. The returned tuple can be used to index the ndarrays
    containing paths, offsets, and lengths in ManifestArrays.

    Parameters
    ----------
    key
        The key in the Zarr store to parse.
    separator
        The chunk key separator used in the Zarr store.
    expand_pattern
        Whether to expand the pattern matching to include /c to protect against
        group structures that look like chunks

    Returns
    -------
    tuple containing chunk indexes.

    Raises
    ------
    ValueError
        If the key does not match the expected node structure for a chunk according the
        [Zarr V3 specification][https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/index.html].

    """
    # Keys ending in `/c` are scalar arrays. The paths, offsets, and lengths in a chunk manifest
    # of a scalar array should also be scalar arrays that can be indexed with an empty tuple.
    if key.endswith("/c") or key == "c":
        return ()

    pattern = compiled_chunk_pattern(separator, expand=expand_pattern)

    if not (match := re.search(pattern, key)):
        msg = (
            f"Key {key!r} with separator {separator!r} did not match the "
            "expected pattern for nodes in the Zarr hierarchy."
        )
        raise ValueError(msg)

    chunk_component = match.group().removeprefix("/").removeprefix(f"c{separator}")
    return tuple(int(ind) for ind in chunk_component.split(separator))


def construct_chunk_pattern(
    separator: ChunkKeySeparator, *, expand: bool = False
) -> str:
    """
    Produce a pattern for finding chunk indices from a key within a Zarr store
    using [re.match][] or [re.search][].

    Parameters
    ----------
    separator
        The chunk key separator used in the Zarr store.

    Returns
    -------
    String representation of regular expression for a chunk key index.

    Raises
    ------
    ValueError
        If `separator` is not a valid separator as specified by the type
        `ChunkKeySeparator`.
    """
    if separator not in CHUNK_KEY_SEPARATORS:
        msg = f"chunk key separator must be one of {CHUNK_KEY_SEPARATORS}: {separator}"
        raise ValueError(msg)

    # Surround separator in square brackets to ensure it's an exact character match.
    sep_pattern = rf"[{separator}]"
    # Matches exactly "0" or an unsigned integer that does not begin with zero
    integer_pattern = r"(?:[1-9]\d*|0)"
    # Matches exactly "c" or an integer followed by zero or more integers each
    # separated by a valid chunk key separator (e.g., a period).
    key_pattern = rf"(?:c|{integer_pattern}(?:{sep_pattern}{integer_pattern})*)$"

    # If expand=True, allow key to start with "c<separator>" or "/c<separator>".
    # In the former case, a full match is performed.
    return rf"(?:^|/)c{sep_pattern}{key_pattern}" if expand else key_pattern


@functools.cache
def compiled_chunk_pattern(
    separator: ChunkKeySeparator, *, expand: bool = False
) -> re.Pattern:
    """
    Produce a pattern for finding chunk indices from a key within a Zarr store
    using [re.match][] or [re.search][].

    Parameters
    ----------
    separator
        The chunk key separator used in the Zarr store.

    Returns
    -------
    Regular expression Pattern for a chunk key index.

    Raises
    ------
    ValueError
        If `separator` is not a valid separator as specified by the type
        `ChunkKeySeparator`.

    Notes
    -----
    This function simply calls [construct_chunk_pattern][], compiles the result,
    caches it, and returns the cached pattern for performance.  See
    [contruct_chunk_pattern][] for examples.
    """
    return re.compile(construct_chunk_pattern(separator, expand=expand))


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


def _remove_elements_at_positions(
    t: tuple[int, ...], pos: list[int]
) -> tuple[int, ...]:
    return tuple(x for i, x in enumerate(t) if i not in pos)


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


def check_same_shapes_except_axes(
    shapes: list[tuple[int, ...]], except_axes: list[int]
):
    """Check that shapes are compatible for concatenation"""

    shapes_without_axes = [
        _remove_elements_at_positions(shape, except_axes) for shape in shapes
    ]

    first_shape, *other_shapes = shapes_without_axes
    for other_shape in other_shapes:
        if other_shape != first_shape:
            raise ValueError(
                f"Cannot concatenate arrays with shapes {[shape for shape in shapes]}"
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
    check_same_chunk_shapes([manifest_chunk_shape(arr.metadata) for arr in arrays])


def check_compatible_arrays(
    ma: "ManifestArray", existing_array: "Array", append_axis: int
):
    check_combinable_zarr_arrays([ma, existing_array])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def manifest_chunk_shape(
    metadata: Union[ArrayV3Metadata, "ArrayV2Metadata"],
) -> tuple[int, ...]:
    """
    The shape of the region of the array that one chunk manifest entry locates.

    For a sharded array this is the *shard* shape, because each manifest entry points at
    a whole shard. Note this is deliberately not ``ArrayV3Metadata.chunks``, which zarr
    defines as the shape of an *inner* chunk when a sharding codec is present.

    Parameters
    ----------
    metadata
        Metadata of the array whose manifest unit is wanted. Zarr V2 metadata is accepted
        because [check_combinable_zarr_arrays][virtualizarr.manifests.utils.check_combinable_zarr_arrays]
        may be handed a V2 `zarr.Array` alongside `ManifestArray`s.

    Returns
    -------
    The shape covered by one manifest entry: the shard shape if `metadata` has a sharding
    codec, else the chunk shape.
    """
    if not isinstance(metadata, ArrayV3Metadata):
        # Zarr V2 has no sharding, so a chunk is the manifest's unit
        return tuple(metadata.chunks)
    return tuple(cast("RegularChunkGridMetadata", metadata.chunk_grid).chunk_shape)


def _realign_inner_chunk_shape(
    old_chunks: tuple[int, ...],
    new_chunks: tuple[int, ...],
    old_inner_chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Carry a shard's inner chunk shape across a change of axes in the outer chunk shape.

    An outer chunk shape can only ever change by adding or removing length-1 axes - a
    chunk covers a fixed number of array elements, so any other change would describe
    different chunks. Adding an axis (`stack`, `expand_dims`, `broadcast_to`) gives the
    inner shape a 1 in the same place; removing one (integer indexing, which is only
    legal where the chunk length is already 1) drops the corresponding inner axis. Both
    leave the inner chunk grid, the C-order shard index layout, and every inner chunk's
    bytes untouched, so the encoded shard stays byte-for-byte valid.

    Where ``old_chunks`` already contains 1s the alignment is ambiguous, but harmlessly
    so: an ambiguous position holds a 1 in the outer shape, so the inner shape must hold
    a 1 there too (it divides the outer), and inserting or dropping a 1 on either side of
    an existing 1 gives the same result.

    Parameters
    ----------
    old_chunks
        The outer chunk shape before the change of axes, i.e. one shard.
    new_chunks
        The outer chunk shape after the change of axes. Must be `old_chunks` with length-1
        axes added and/or removed.
    old_inner_chunks
        The shard's inner chunk shape, on `old_chunks`' axes. Same length as `old_chunks`,
        and divides it element-wise.

    Returns
    -------
    `old_inner_chunks` mapped onto `new_chunks`' axes, with a 1 at each added axis and the
    entry at each dropped axis removed. Same length as `new_chunks`.

    Raises
    ------
    ValueError
        If `new_chunks` is not `old_chunks` with length-1 axes added or removed, so no
        alignment exists.
    """
    new_inner_chunks: list[int] = []
    old_axis = new_axis = 0
    while old_axis < len(old_chunks) or new_axis < len(new_chunks):
        if (
            old_axis < len(old_chunks)
            and new_axis < len(new_chunks)
            and old_chunks[old_axis] == new_chunks[new_axis]
        ):
            new_inner_chunks.append(old_inner_chunks[old_axis])
            old_axis += 1
            new_axis += 1
        elif new_axis < len(new_chunks) and new_chunks[new_axis] == 1:
            new_inner_chunks.append(1)  # axis added
            new_axis += 1
        elif old_axis < len(old_chunks) and old_chunks[old_axis] == 1:
            old_axis += 1  # axis dropped
        else:
            raise ValueError(
                f"chunk shape {new_chunks} is not {old_chunks} with length-1 axes added "
                "or removed, so it cannot describe the same chunks"
            )
    return tuple(new_inner_chunks)


def _realign_sharding_codecs(
    codecs: Iterable[dict[str, Any]],
    old_chunks: tuple[int, ...],
    new_chunks: tuple[int, ...],
) -> list[dict[str, Any]]:
    """
    Realign the inner ``chunk_shape`` of any ``sharding_indexed`` codec onto new axes.

    Zarr requires a shard's inner chunk shape to have the same number of dimensions as
    the array, so changing an array's axes means changing the shard config to match.

    Parameters
    ----------
    codecs
        A codec pipeline, as plain dicts (i.e. straight out of `ArrayV3Metadata.to_dict`).
        Codecs other than `sharding_indexed` are passed through untouched.
    old_chunks
        The chunk shape these codecs currently encode, on the old axes. For a top-level
        pipeline this is the array's outer chunk shape; when recursing into a nested shard
        it is the enclosing shard's inner chunk shape.
    new_chunks
        The same shape on the new axes, as `old_chunks` with length-1 axes added and/or
        removed.

    Returns
    -------
    A new codec list. Sharding codecs are deep-copied with their inner `chunk_shape`
    realigned, so the input dicts are never mutated.
    """
    updated = []
    for codec in codecs:
        if codec.get("name") != "sharding_indexed":
            updated.append(codec)
            continue

        codec = copy.deepcopy(codec)
        configuration = codec["configuration"]
        old_inner_chunks = tuple(configuration["chunk_shape"])
        new_inner_chunks = _realign_inner_chunk_shape(
            old_chunks, new_chunks, old_inner_chunks
        )
        configuration["chunk_shape"] = new_inner_chunks
        # shards may themselves contain shards, for which this shard's inner chunk shape
        # is in turn the outer one
        configuration["codecs"] = _realign_sharding_codecs(
            configuration["codecs"], old_inner_chunks, new_inner_chunks
        )
        updated.append(codec)
    return updated


def copy_and_replace_metadata(
    old_metadata: ArrayV3Metadata,
    new_shape: list[int] | None = None,
    new_chunks: list[int] | None = None,
    new_dimension_names: Iterable[str] | None | Literal["default"] = "default",
    new_attributes: dict | None = None,
) -> ArrayV3Metadata:
    """
    Update metadata to reflect a new shape and/or chunk shape.

    Parameters
    ----------
    old_metadata
        Metadata to copy from. Never mutated.
    new_shape
        Replacement array shape, or None to keep the existing one.
    new_chunks
        Replacement *outer* chunk shape - the shard shape for a sharded array, see
        [manifest_chunk_shape][virtualizarr.manifests.utils.manifest_chunk_shape] - or None
        to keep the existing one. If this changes the number of axes, any sharding codec's
        inner `chunk_shape` is realigned onto the new axes to match; the change must then
        be length-1 axes added and/or removed, since a chunk covers a fixed number of
        elements.
    new_dimension_names
        Replacement dimension names, or None to clear them. Defaults to the sentinel
        `"default"`, meaning leave them as they are - None cannot serve as that sentinel
        because it is itself a valid value for zarr's `dimension_names`.
    new_attributes
        Replacement attributes dict, or None to keep the existing one.

    Returns
    -------
    New metadata with the requested replacements applied.

    Raises
    ------
    ValueError
        If `new_chunks` changes the number of axes by anything other than adding or
        removing length-1 axes, leaving a sharding codec's inner chunk shape unalignable.
    """
    # TODO this should really be upstreamed into zarr-python

    metadata_copy = old_metadata.to_dict().copy()

    if new_shape is not None:
        metadata_copy["shape"] = parse_shapelike(new_shape)  # type: ignore[assignment]
    if new_chunks is not None:
        old_chunks = manifest_chunk_shape(old_metadata)
        new_chunks = list(new_chunks)
        metadata_copy["chunk_grid"] = {
            "name": "regular",
            "configuration": {"chunk_shape": tuple(new_chunks)},
        }
        if len(new_chunks) != len(old_chunks):
            # a sharding codec's inner chunk_shape must match the array's ndim, so it has
            # to gain or lose the same length-1 axes the outer chunk shape just did
            metadata_copy["codecs"] = _realign_sharding_codecs(
                cast(list[dict[str, Any]], metadata_copy["codecs"]),
                old_chunks,
                tuple(new_chunks),
            )
    if new_dimension_names != "default":
        # need the option to use the literal string "default" as a sentinel value because None is a valid choice for zarr dimension_names
        metadata_copy["dimension_names"] = parse_dimension_names(new_dimension_names)
    if new_attributes is not None:
        metadata_copy["attributes"] = new_attributes

    # ArrayV3Metadata.from_dict removes extra keys zarr_format and node_type
    new_metadata = ArrayV3Metadata.from_dict(metadata_copy)
    return new_metadata
