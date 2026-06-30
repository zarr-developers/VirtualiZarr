from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, cast

import numpy as np
from zarr.codecs import BytesCodec, TransposeCodec
from zarr.core.metadata.v3 import ArrayV3Metadata

from virtualizarr.manifests.array_api import expand_dims
from virtualizarr.manifests.manifest import ChunkManifest
from virtualizarr.manifests.utils import copy_and_replace_metadata

# indexer with only basic selectors, no new axes or ellipsis
T_BasicIndexer_1d: TypeAlias = int | slice | np.ndarray
# indexer allowing new axes but without ellipses
T_SimpleIndexer_1d: TypeAlias = T_BasicIndexer_1d | None
# general valid indexer representing any possible user input
T_Indexer_1d: TypeAlias = T_SimpleIndexer_1d | EllipsisType
T_Indexer: TypeAlias = T_Indexer_1d | tuple[T_Indexer_1d, ...]


class SubChunkIndexingError(ValueError):
    """
    Raised when an indexer would split individual chunks of a compressed ManifestArray.
    """


if TYPE_CHECKING:
    from virtualizarr.manifests.array import ManifestArray


def index(marr: "ManifestArray", indexer: T_Indexer) -> "ManifestArray":
    """Index into a ManifestArray"""
    indexer_tuple = check_and_sanitize_indexer_type(indexer)
    indexer_without_ellipsis = check_shape_and_maybe_replace_ellipsis(
        indexer_tuple, marr.ndim
    )
    return apply_indexer(marr, indexer_without_ellipsis)


def check_and_sanitize_indexer_type(key: T_Indexer) -> tuple[T_Indexer_1d, ...]:
    """Check for invalid input types, and narrow the return type to a tuple of valid 1D indexers."""
    if isinstance(key, (int, slice, EllipsisType, np.ndarray)) or key is None:
        indexer = cast(tuple[T_Indexer_1d, ...], (key,))
    elif isinstance(key, tuple):
        for dim_indexer in key:
            if (
                not isinstance(dim_indexer, (int, slice, np.ndarray))
                and dim_indexer is not None
                and dim_indexer is not ...
            ):
                raise TypeError(
                    f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
                )
        indexer = cast(tuple[T_Indexer_1d, ...], key)
    else:
        raise TypeError(
            f"indexer must be of type int, slice, ellipsis, None, or np.ndarray; or a tuple of such types. Got {key}"
        )
    return indexer


def check_shape_and_maybe_replace_ellipsis(
    indexer: tuple[T_Indexer_1d, ...], arr_ndim: int
) -> tuple[T_SimpleIndexer_1d, ...]:
    """Deal with any ellipses, potentially by expanding the indexer to match the shape of the array."""

    num_single_axis_indexing_expressions = len(
        [ind_1d for ind_1d in indexer if ind_1d is not None and ind_1d is not ...]
    )
    num_ellipses = len([ind_1d for ind_1d in indexer if ind_1d is ...])

    if num_ellipses > 1:
        raise ValueError(
            f"Invalid indexer. Indexers containing multiple Ellipses are invalid, but indexer={indexer} contains {num_ellipses} ellipses"
        )

    bad_shape_error_msg = (
        "Invalid indexer for array. Indexer must contain a number of single-axis indexing expressions less than or equal to the length of the array. "
        f"However indexer={indexer} has {num_single_axis_indexing_expressions} single-axis indexing expressions and array has {arr_ndim} dimensions."
        "\nIf concatenating using xarray, ensure all non-coordinate data variables to be concatenated include the concatenation dimension, "
        "or consider passing `data_vars='minimal'` and `coords='minimal'` to the xarray combining function."
    )

    if num_ellipses == 1:
        if num_single_axis_indexing_expressions > arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return replace_single_ellipsis(
                indexer, arr_ndim, num_single_axis_indexing_expressions
            )
    else:  # num_ellipses == 0
        if num_single_axis_indexing_expressions != arr_ndim:
            raise ValueError(bad_shape_error_msg)
        else:
            return cast(tuple[T_SimpleIndexer_1d, ...], indexer)


def replace_single_ellipsis(
    indexer: tuple[T_Indexer_1d, ...],
    arr_ndim: int,
    num_single_axis_indexing_expressions: int,
) -> tuple[T_SimpleIndexer_1d, ...]:
    """
    Replace ellipsis with 0 or more slice(None)s until there are ndim single-axis indexing expressions (so ignoring Nones).
    """
    indexer_as_list = list(indexer)

    # find this position before modifying in-place
    position_of_ellipsis = indexer_as_list.index(...)

    # need to remove the ellipsis separate from replacement, because the ellipsis may have been totally superfluous already,
    # and it still needs to be removed even if no further replacement will occur
    indexer_as_list.remove(...)

    # replace ellipsis with the equivalent number of no-op slices
    num_extra_slices_needed = arr_ndim - num_single_axis_indexing_expressions
    new_slices = [slice(None)] * num_extra_slices_needed

    # insert multiple elements into one position
    indexer_as_list[position_of_ellipsis:position_of_ellipsis] = new_slices

    return cast(tuple[T_SimpleIndexer_1d, ...], tuple(indexer_as_list))


def apply_indexer(
    marr: "ManifestArray", indexer: tuple[T_SimpleIndexer_1d, ...]
) -> "ManifestArray":
    """
    Apply the simplified indexer to all dimensions of the array.

    Iterates over the axes and applies each indexer one-by-one.
    Encountering a None means inserting a new axis at that position.
    """
    # handles composition of subsetting and expanding dims by following the two-step approach described in https://github.com/data-apis/array-api/pull/408#issuecomment-1091056873

    indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...] = tuple(
        [ind_1d for ind_1d in indexer if ind_1d is not None]
    )
    output_arr = apply_selection(marr, indexer_without_newaxes)

    for position, axis_indexer in enumerate(indexer):
        if axis_indexer is None:
            output_arr = expand_dims(output_arr, axis=position)

    return output_arr


def _is_reindex_indexer(indexer_1d: T_BasicIndexer_1d) -> TypeGuard[np.ndarray]:
    """True if this 1D indexer is an integer array (xarray's reindex indexer)."""
    return isinstance(indexer_1d, np.ndarray) and indexer_1d.dtype.kind in ("i", "u")


def _collapse_outer_indexer(
    indexer_1d: T_BasicIndexer_1d, axis: int
) -> T_BasicIndexer_1d:
    """Collapse a broadcast reindex indexer back to its 1D per-axis form.

    When xarray reindexes more than one dimension at once it sends a single
    *broadcast* (vectorized) indexer: one integer array per axis, shaped to the
    array's rank with the real labels on its own axis and length 1 everywhere
    else (e.g. ``(N, 1)`` for axis 0 and ``(1, M)`` for axis 1). Because reindex
    indexing is orthogonal, each such array reshapes losslessly to the 1D
    indexer for ``axis``. int/slice indexers and already-1D arrays pass through.
    """
    if not (isinstance(indexer_1d, np.ndarray) and indexer_1d.ndim > 1):
        return indexer_1d
    if any(size != 1 for ax, size in enumerate(indexer_1d.shape) if ax != axis):
        raise NotImplementedError(
            "Pointwise (vectorized) fancy indexing is not supported on a "
            f"ManifestArray; received a {indexer_1d.shape} indexer for axis {axis}."
        )
    return indexer_1d.reshape(-1)


def _is_full_slice(indexer_1d: T_BasicIndexer_1d, length: int) -> bool:
    """True if this slice selects the whole axis (a no-op)."""
    if not isinstance(indexer_1d, slice):
        return False
    return (
        indexer_1d == slice(None)
        or indexer_1d == slice(0, length, None)
        or (indexer_1d.indices(length) == (0, length, 1))
    )


def _apply_reindex(
    marr: "ManifestArray", indexers: tuple[T_BasicIndexer_1d, ...]
) -> "ManifestArray":
    """
    Remap the chunk grid for one or more integer-array (reindex) indexers.

    Each integer-array axis is reindexed by translating its indexer into a
    chunk-grid map (null chunks for missing positions). Any other axis must be a
    full-axis slice (the shape xarray's reindex/alignment produces); combining
    reindexing with chunk subsetting in a single call is not supported.
    """
    from virtualizarr.manifests.reindex import chunk_map_from_indexer

    result = marr
    for axis, raw_indexer_1d in enumerate(indexers):
        indexer_1d = _collapse_outer_indexer(raw_indexer_1d, axis)
        if _is_reindex_indexer(indexer_1d):
            chunk_size = result.metadata.chunks[axis]
            try:
                chunk_map = chunk_map_from_indexer(
                    np.asarray(indexer_1d), chunk_size, result.shape[axis]
                )
            except NotImplementedError as err:
                # chunk_map_from_indexer is reached from deep inside xarray's
                # reindex/alignment machinery, so re-raise with call-site context
                # naming the operation, the axis, and what the user can do.
                raise NotImplementedError(
                    f"Cannot align/reindex this virtual array along axis {axis} "
                    f"(chunk size {chunk_size}) without materializing data: the "
                    "target coordinate labels do not line up with chunk boundaries, "
                    "so filling them would require splitting a chunk. VirtualiZarr "
                    "only supports whole-chunk appends, inserts, and reorders. "
                    "Re-chunk along this dimension, or load the variable into memory, "
                    "if you need this alignment. See "
                    "https://github.com/zarr-developers/VirtualiZarr/issues/51."
                ) from err
            result = result._reindex_axis(axis, chunk_map, new_size=len(indexer_1d))
        elif _is_full_slice(indexer_1d, marr.shape[axis]):
            continue
        else:
            raise NotImplementedError(
                "Combining reindexing (an integer-array indexer) with chunk "
                "subsetting in a single operation is not supported."
            )
    return result


def apply_selection(
    marr: "ManifestArray", indexer_without_newaxes: tuple[T_BasicIndexer_1d, ...]
) -> "ManifestArray":
    """
    Apply chunk-aligned subsetting along each dimension.

    Slices and integer indexers are only supported if they align with chunk boundaries, since
    splitting individual chunks would require loading their bytes. See GitHub issue #51.

    Integer indexers drop the indexed axis (numpy / array-API semantics); slice indexers
    preserve it. Integer indexing is only legal when ``chunk_size == 1`` along that axis,
    since picking a single element of a larger chunk would require splitting it.
    """
    from virtualizarr.manifests.array import ManifestArray

    # at this point there should be no ellipsis, no Nones, and one 1D indexer for each axis.
    assert len(indexer_without_newaxes) == marr.ndim

    # An integer array indexer is xarray's reindex/alignment indexer (with -1
    # marking missing labels). Rather than reject it as fancy indexing, remap the
    # chunk grid: missing positions become null-path chunks that read back as
    # fill_value. This is what lets xarray's reindex/align/concat machinery work
    # over ManifestArrays without materializing data.
    if any(_is_reindex_indexer(ind) for ind in indexer_without_newaxes):
        return _apply_reindex(marr, indexer_without_newaxes)

    # validate types and reject anything we can never support per-axis
    narrowed_indexers: list[int | slice] = []
    for indexer_1d in indexer_without_newaxes:
        if isinstance(indexer_1d, np.ndarray):
            raise NotImplementedError(
                f"Unsupported indexer. So-called 'fancy indexing' via numpy arrays is not supported, but received {indexer_1d}"
            )
        if not isinstance(indexer_1d, (int, slice)):
            raise TypeError(f"Invalid indexer type: {indexer_1d}")
        narrowed_indexers.append(indexer_1d)

    sub_chunk_axis = _uncompressed_sub_chunk_axis(marr.metadata)

    new_shape: list[int] = []
    new_chunks: list[int] = []
    chunk_grid_selectors: list[int | slice] = []
    kept_axes: list[int] = []
    # At most one sub-chunk axis (whichever axis has the largest byte stride in storage).
    # The byte adjustment is uniform across every surviving chunk, since chunks share layout.
    sub_chunk_byte_adjust: tuple[int, int] | None = None
    for axis, (axis_length, chunk_size, indexer_1d) in enumerate(
        zip(marr.shape, marr.metadata.chunks, narrowed_indexers, strict=True)
    ):
        chunk_grid_selector: int | slice
        if axis == sub_chunk_axis and _is_sub_chunk_slice(
            indexer_1d, axis_length, chunk_size
        ):
            chunk_grid_selector, new_axis_length, sub_chunk_byte_adjust = (
                _compute_sub_chunk_axis_selection(
                    indexer_1d,
                    axis_length=axis_length,
                    chunk_size=chunk_size,
                    other_axis_chunks=tuple(
                        c for i, c in enumerate(marr.metadata.chunks) if i != axis
                    ),
                    itemsize=marr.dtype.itemsize,
                )
            )
            new_chunks_for_axis = new_axis_length
        else:
            chunk_grid_selector, new_axis_length = _compute_chunk_aligned_selection_1d(
                indexer_1d, axis_length=axis_length, chunk_size=chunk_size
            )
            new_chunks_for_axis = chunk_size

        chunk_grid_selectors.append(chunk_grid_selector)
        # int indexers drop the axis from the output array; slices preserve it (including the
        # sub-chunk path, which uses a length-1 chunk-grid slice selector).
        if not isinstance(indexer_1d, int):
            new_shape.append(new_axis_length)
            new_chunks.append(new_chunks_for_axis)
            kept_axes.append(axis)

    chunk_grid_selectors_tuple = tuple(chunk_grid_selectors)

    # short-circuit if every axis selects the whole chunk grid via a slice (a no-op).
    # A pending sub-chunk byte adjustment is real work even if its single source chunk
    # happens to span the whole chunk grid along that axis, so don't short-circuit then.
    if sub_chunk_byte_adjust is None and all(
        isinstance(cgs, slice) and cgs == slice(0, dim, 1)
        for cgs, dim in zip(chunk_grid_selectors_tuple, marr.manifest.shape_chunk_grid)
    ):
        return marr

    new_manifest = _subset_manifest(marr.manifest, chunk_grid_selectors_tuple)
    if sub_chunk_byte_adjust is not None:
        new_manifest = _shift_manifest_byte_ranges(new_manifest, *sub_chunk_byte_adjust)
    old_dimension_names = marr.metadata.dimension_names
    # zarr's dimension_names is tuple[str | None, ...] but copy_and_replace_metadata's
    # type hint says Iterable[str]; the runtime handles None entries fine, so cast through.
    new_dimension_names: Any
    if old_dimension_names is None:
        new_dimension_names = "default"  # sentinel: leave as None
    else:
        new_dimension_names = tuple(old_dimension_names[a] for a in kept_axes)
    new_metadata = copy_and_replace_metadata(
        marr.metadata,
        new_shape=new_shape,
        new_chunks=new_chunks,
        new_dimension_names=new_dimension_names,
    )
    return ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)


def _compute_chunk_aligned_selection_1d(
    indexer_1d: int | slice, axis_length: int, chunk_size: int
) -> tuple[int | slice, int]:
    """
    Translate a 1D array-space indexer (int or slice) into a chunk-grid selector plus the
    new axis length. The selector is an ``int`` for int indexers (so the chunk-grid axis
    is dropped) and a ``slice`` for slice indexers (so the chunk-grid axis is preserved).

    Raises SubChunkIndexingError if the selection would require splitting individual chunks.
    """
    if isinstance(indexer_1d, int):
        i = indexer_1d
        if i < 0:
            # Allow negative indexing - this makes it wrap around
            i += axis_length
        if not (0 <= i < axis_length):
            raise IndexError(
                f"index {indexer_1d} is out of bounds for axis with size {axis_length}"
            )
        start, stop, step = i, i + 1, 1
    else:
        start, stop, step = indexer_1d.indices(axis_length)

    # The final chunk may legitimately be partial, so allow stop == axis_length even if
    # axis_length % chunk_size != 0; otherwise both endpoints must land on chunk boundaries.
    # TODO step != 1 we can actually support for uncompressed arrays along the first axis,
    # see https://github.com/zarr-developers/VirtualiZarr/issues/86
    if (
        step != 1
        or start % chunk_size != 0
        or (stop != axis_length and stop % chunk_size != 0)
    ):
        raise SubChunkIndexingError(
            f"Cannot index ManifestArray axis of length {axis_length} and chunk length "
            f"{chunk_size} with {indexer_1d!r}: slice would split individual chunks, "
            "which a ManifestArray cannot do without loading the underlying data."
        )

    chunk_start = start // chunk_size

    if isinstance(indexer_1d, int):
        # int indexer drops the array axis, so the chunk-grid axis is dropped too via an int selector
        return chunk_start, 1

    # slice indexer: ceil-divide stop so a partial final chunk is included when stop == axis_length
    chunk_stop = -(-stop // chunk_size)
    return slice(chunk_start, chunk_stop, 1), stop - start


def _compute_sub_chunk_axis_selection(
    indexer_1d: slice,
    axis_length: int,
    chunk_size: int,
    other_axis_chunks: tuple[int, ...],
    itemsize: int,
) -> tuple[slice, int, tuple[int, int]]:
    """
    Translate a sub-chunk slice along the eligible (largest-stride) storage axis into a
    chunk-grid selector, an output axis length, and a uniform byte adjustment
    ``(offset_delta, new_chunk_byte_length)`` applied to every surviving chunk reference.

    Callers must have already confirmed that this slice is sub-chunk-eligible via
    ``_is_sub_chunk_slice`` and that the array is uncompressed via
    ``_uncompressed_sub_chunk_axis``.
    """
    start, stop, _ = indexer_1d.indices(axis_length)
    chunk_index = start // chunk_size
    new_axis_length = stop - start
    # Bytes per index step along this axis within one chunk is the product of every
    # *other* axis's chunk size, times itemsize. Order doesn't matter since the product
    # is commutative.
    stride_bytes = int(np.prod(other_axis_chunks)) * itemsize
    inner_offset_bytes = (start - chunk_index * chunk_size) * stride_bytes
    sub_chunk_byte_adjust = (inner_offset_bytes, new_axis_length * stride_bytes)
    return (
        slice(chunk_index, chunk_index + 1, 1),
        new_axis_length,
        sub_chunk_byte_adjust,
    )


def _subset_manifest(
    manifest: ChunkManifest, chunk_grid_selectors: tuple[int | slice, ...]
) -> ChunkManifest:
    """
    Subset a ChunkManifest by indexing its underlying chunk-grid arrays. Each entry of
    ``chunk_grid_selectors`` is either an int (which drops the corresponding chunk-grid axis)
    or a slice (which keeps it).
    """
    # When every axis is int-indexed, numpy returns a 0D scalar (Python str for StringDType,
    # numpy scalar for the numeric arrays). Wrap back into a 0D ndarray so from_arrays accepts it.
    # np.asarray's return type erases the dtype param, so cast back to keep from_arrays happy.
    new_paths = cast(
        "np.ndarray[Any, np.dtypes.StringDType]",
        np.asarray(manifest._paths[chunk_grid_selectors], dtype=manifest._paths.dtype),
    )
    new_offsets = cast(
        "np.ndarray[Any, np.dtype[np.uint64]]",
        np.asarray(
            manifest._offsets[chunk_grid_selectors], dtype=manifest._offsets.dtype
        ),
    )
    new_lengths = cast(
        "np.ndarray[Any, np.dtype[np.uint64]]",
        np.asarray(
            manifest._lengths[chunk_grid_selectors], dtype=manifest._lengths.dtype
        ),
    )

    if manifest._inlined:
        # For each old chunk-grid key, keep it only if int-indexed axes match exactly and
        # slice-indexed axes fall inside the slice. Re-map the surviving key by omitting
        # int-indexed positions and offsetting slice-indexed positions by the slice start.
        new_inlined: dict[tuple[int, ...], bytes] = {}
        for coords, data in manifest._inlined.items():
            new_coord: list[int] = []
            keep = True
            for coord, sel in zip(coords, chunk_grid_selectors):
                if isinstance(sel, int):
                    if coord != sel:
                        keep = False
                        break
                else:
                    if not (sel.start <= coord < sel.stop):
                        keep = False
                        break
                    new_coord.append(coord - sel.start)
            if keep:
                new_inlined[tuple(new_coord)] = data
    else:
        new_inlined = {}

    return ChunkManifest.from_arrays(
        paths=new_paths,
        offsets=new_offsets,
        lengths=new_lengths,
        inlined=new_inlined,
        validate_paths=False,
    )


def _uncompressed_sub_chunk_axis(metadata: ArrayV3Metadata) -> int | None:
    """
    Return the axis along which sub-chunk slicing is implementable for this array, or
    ``None`` if the codec stack disqualifies it.

    Sub-chunk slicing rewrites an existing chunk reference's byte offset and length,
    so it only works when chunk bytes are raw element values in a fixed memory order —
    i.e., no compression, no value transforms, no checksums. The eligible codec stacks
    are:

    - ``[BytesCodec]`` — C-order layout; the axis with the largest byte stride is axis 0.
    - ``[TransposeCodec(order=perm), BytesCodec]`` — stored layout is the logical array
      permuted by ``perm``; the axis with the largest byte stride in storage is logical
      axis ``perm[0]``. For the F-order case ``perm = (n-1, n-2, ..., 0)`` this picks out
      the last axis.
    """
    codecs = metadata.codecs
    if len(codecs) == 1 and isinstance(codecs[0], BytesCodec):
        return 0
    if (
        len(codecs) == 2
        and isinstance(codecs[0], TransposeCodec)
        and isinstance(codecs[1], BytesCodec)
    ):
        return int(codecs[0].order[0])
    return None


def _is_sub_chunk_slice(
    indexer_1d: int | slice, axis_length: int, chunk_size: int
) -> TypeGuard[slice]:
    """
    True iff this is a slice that should take the sub-chunk path: step == 1, non-empty,
    fits entirely within one source chunk, and is NOT already chunk-aligned (chunk-aligned
    slices go through the simpler aligned path).

    Typed as ``TypeGuard[slice]`` so callers can pass the narrowed indexer straight into
    helpers that take a ``slice``.
    """
    if not isinstance(indexer_1d, slice):
        return False
    start, stop, step = indexer_1d.indices(axis_length)
    if step != 1 or start >= stop:
        return False
    # chunk-aligned slices are handled by _compute_chunk_aligned_selection_1d
    aligned = start % chunk_size == 0 and (
        stop == axis_length or stop % chunk_size == 0
    )
    if aligned:
        return False
    # contained in a single source chunk?
    return start // chunk_size == (stop - 1) // chunk_size


def _shift_manifest_byte_ranges(
    manifest: ChunkManifest, offset_delta: int, new_length: int
) -> ChunkManifest:
    """
    Return a new ``ChunkManifest`` whose virtual chunk references point to a uniform
    sub-range of each original chunk: ``offset += offset_delta`` and ``length = new_length``.

    Used by the uncompressed-axis-0 sub-chunk path, where every surviving chunk shares the
    same byte layout and therefore the same byte adjustment.
    """
    new_offsets = cast(
        "np.ndarray[Any, np.dtype[np.uint64]]",
        manifest._offsets + np.uint64(offset_delta),
    )
    new_lengths = cast(
        "np.ndarray[Any, np.dtype[np.uint64]]",
        np.full_like(manifest._lengths, np.uint64(new_length)),
    )
    # paths and any inlined-chunk dict carry through unchanged: inlined chunks aren't
    # involved here (this path is only taken for uncompressed virtual references).
    return ChunkManifest.from_arrays(
        paths=manifest._paths,
        offsets=new_offsets,
        lengths=new_lengths,
        inlined=dict(manifest._inlined),
        validate_paths=False,
    )
