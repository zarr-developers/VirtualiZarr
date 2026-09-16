import itertools
from typing import TYPE_CHECKING, Any, Callable, Union, cast

import numpy as np
import zarr
from zarr.experimental import ChunkGrid

from .manifest import MISSING_CHUNK_PATH, ChunkManifest
from .utils import (
    check_combinable_zarr_arrays,
    check_no_partial_chunks_on_concat_axis,
    check_same_ndims,
    check_same_shapes,
    check_same_shapes_except_on_concat_axis,
    chunk_grid_sizes,
    copy_and_replace_metadata,
    full_chunk_edges,
    manifest_chunk_shape,
)

if TYPE_CHECKING:
    from .array import ManifestArray


MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS: dict[
    str, Callable
] = {}  # populated by the @implements decorators below


def implements(numpy_function):
    """Register an __array_function__ implementation for ManifestArray objects."""

    def decorator(func):
        MANIFESTARRAY_HANDLED_ARRAY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.result_type)
def result_type(*arrays_and_dtypes: Union["ManifestArray", np.dtype]) -> np.dtype:
    """
    Resolve a result dtype for ManifestArray arguments.

    Called by xarray both to check that concat/stack inputs share a dtype and,
    during reindex/alignment, to combine a ManifestArray with a scalar fill
    value. A ManifestArray's dtype is fixed by its metadata (the same metadata
    that defines its fill value), so when exactly one ManifestArray is combined
    with scalars/dtypes we return its dtype rather than promoting — keeping the
    array's native dtype and declared fill value through a reindex.
    """
    from virtualizarr.manifests.array import ManifestArray

    manifest_dtypes = [
        obj.dtype for obj in arrays_and_dtypes if isinstance(obj, ManifestArray)
    ]
    if len(manifest_dtypes) == 1:
        return manifest_dtypes[0]

    dtypes = [
        obj.dtype if isinstance(obj, ManifestArray) else np.dtype(obj)
        for obj in arrays_and_dtypes
    ]
    first_dtype, *other_dtypes = dtypes
    unique_dtypes = set(dtypes)
    for other_dtype in other_dtypes:
        if other_dtype != first_dtype:
            raise ValueError(
                f"Cannot combine arrays with inconsistent dtypes, but got {len(unique_dtypes)} distinct dtypes: {unique_dtypes}"
            )

    return first_dtype


@implements(np.where)
def where(condition, x, y, /):
    """
    Support xarray's reindex/alignment fill, which calls
    ``where(~mask, gathered_array, fill_value)`` after gathering chunks.

    The gathered ManifestArray already carries null-path chunks (which read back
    as ``fill_value``) at exactly the missing positions, so this is an identity
    whenever the requested fill positions coincide with the array's missing
    chunks. In that case ``x`` is returned unchanged and the manifest's own fill
    value governs. Any other ``where`` usage (e.g. general boolean masking) would
    require materializing values and is not supported.
    """
    from virtualizarr.manifests.array import ManifestArray

    if isinstance(x, ManifestArray) and np.isscalar(y):
        cond = np.asarray(condition, dtype=bool)
        if cond.shape == x.shape and np.array_equal(~cond, _missing_element_mask(x)):
            return x

    raise NotImplementedError(
        "np.where on a ManifestArray is only supported for the reindex/alignment "
        "fill pattern (filling an array's own missing chunks); general masking "
        "would require materializing values."
    )


def _chunk_sizes(
    arr: "ManifestArray",
) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    """
    Per-axis chunk size(s) of a ManifestArray.

    For a regular grid this is a tuple of ints (e.g. ``(30, 50)``); for a rectilinear
    grid it's a tuple of per-axis chunk-edge tuples (e.g. ``((10, 20, 30), (50, 50))``).

    Deliberately not exposed as ``ManifestArray.chunks`` - xarray's ``is_chunked_array``
    duck-types on ``hasattr(x, "chunks")`` and would misclassify a virtual array as a
    computable dask-like array (see #1016).
    """
    grid = arr.chunk_grid
    return grid.chunk_shape if grid.is_regular else grid.chunk_sizes


def _require_rectilinear_chunks_enabled(context: str) -> None:
    """Raise a clear, actionable error unless rectilinear chunk grids are enabled."""
    if not zarr.config.get("array.rectilinear_chunks"):
        raise ValueError(
            f"{context} would require a rectilinear (variable-length) chunk grid. "
            "Rectilinear chunk grids are an experimental zarr-python feature; enable "
            "them with zarr.config.set({'array.rectilinear_chunks': True}) or the "
            "ZARR_ARRAY__RECTILINEAR_CHUNKS environment variable."
        )


def _missing_element_mask(marr: "ManifestArray") -> np.ndarray:
    """Boolean element-mask (shape == marr.shape), True at missing (null) chunks."""
    mask = marr.manifest._paths == MISSING_CHUNK_PATH
    for axis, chunk_size in enumerate(manifest_chunk_shape(marr.metadata)):
        mask = np.repeat(mask, chunk_size, axis=axis)
    return mask[tuple(slice(0, length) for length in marr.shape)]


@implements(np.concatenate)
def concatenate(
    arrays: tuple["ManifestArray", ...] | list["ManifestArray"],
    /,
    *,
    axis: int | None = 0,
) -> "ManifestArray":
    """
    Concatenate ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.concat`.
    """

    from .array import ManifestArray

    if axis is None:
        raise NotImplementedError(
            "If axis=None the array API requires flattening, which is a reshape, which can't be implemented on a ManifestArray."
        )
    elif not isinstance(axis, int):
        raise TypeError()

    # ensure dtypes, shapes, codecs etc. are consistent
    # (chunk sizes along the concat axis are allowed to differ - that's what a
    # rectilinear chunk grid is for)
    check_combinable_zarr_arrays(arrays, exclude_axis=axis)

    check_same_ndims([arr.ndim for arr in arrays])

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    if axis < 0:
        axis = axis % first_arr.ndim

    arr_shapes = [arr.shape for arr in arrays]
    arr_chunks = [chunk_grid_sizes(arr.metadata) for arr in arrays]
    check_same_shapes_except_on_concat_axis(arr_shapes, axis)
    check_no_partial_chunks_on_concat_axis(arr_shapes, arr_chunks, axis)

    # find what new array shape must be
    new_length_along_concat_axis = sum([shape[axis] for shape in arr_shapes])
    first_shape, *_ = arr_shapes
    new_shape = list(first_shape)
    new_shape[axis] = new_length_along_concat_axis

    # do concatenation of entries in manifest
    concatenated_manifest = _concat_manifests(
        [arr.manifest for arr in arrays], axis=axis
    )

    # The result stays a regular grid only if every input is itself regular and they
    # all declare the same chunk size along the concat axis. Otherwise the concat
    # axis's real per-chunk edges (which may already differ, or may only differ once
    # merged) have to be spelled out explicitly, promoting the result to a rectilinear
    # chunk grid.
    stays_regular = all(arr.chunk_grid.is_regular for arr in arrays) and (
        len({arr.chunk_grid.chunk_shape[axis] for arr in arrays}) == 1
    )

    new_chunks = None
    if not stays_regular:
        _require_rectilinear_chunks_enabled(
            f"Concatenating these arrays along axis {axis}"
        )
        new_chunks = list(full_chunk_edges(first_arr.metadata))
        concat_edges: tuple[int, ...] = ()
        for arr in arrays:
            concat_edges = concat_edges + full_chunk_edges(arr.metadata)[axis]
        new_chunks[axis] = concat_edges

    new_metadata = copy_and_replace_metadata(
        old_metadata=first_arr.metadata, new_shape=new_shape, new_chunks=new_chunks
    )

    return ManifestArray(chunkmanifest=concatenated_manifest, metadata=new_metadata)


@implements(np.stack)
def stack(
    arrays: tuple["ManifestArray", ...] | list["ManifestArray"],
    /,
    *,
    axis: int = 0,
) -> "ManifestArray":
    """
    Stack ManifestArrays by merging their chunk manifests.

    The signature of this function is array API compliant, so that it can be called by `xarray.stack`.
    """

    from .array import ManifestArray

    if not isinstance(axis, int):
        raise TypeError()

    # ensure dtypes, shapes, codecs etc. are consistent
    check_combinable_zarr_arrays(arrays)

    check_same_ndims([arr.ndim for arr in arrays])
    arr_shapes = [arr.shape for arr in arrays]
    check_same_shapes(arr_shapes)

    # Ensure we handle axis being passed as a negative integer
    first_arr = arrays[0]
    if axis < 0:
        axis = axis % first_arr.ndim

    # find what new array shape must be
    length_along_new_stacked_axis = len(arrays)
    first_shape, *_ = arr_shapes
    new_shape = list(first_shape)
    new_shape.insert(axis, length_along_new_stacked_axis)

    # do stacking of entries in manifest
    stacked_manifest = _stack_manifests([arr.manifest for arr in arrays], axis=axis)

    # chunk shape has changed because a new axis has been inserted, with one
    # length-1 chunk per stacked array
    old_chunks = _chunk_sizes(first_arr)
    new_chunks = list(old_chunks)
    # For rectilinear grids, each element is a sequence of edges rather than a
    # single chunk size, so the new axis needs one size-1 edge per stacked array
    if not first_arr.chunk_grid.is_regular:
        _require_rectilinear_chunks_enabled("Stacking these arrays")
        new_chunks.insert(axis, (1,) * length_along_new_stacked_axis)
    else:
        new_chunks.insert(axis, 1)

    new_metadata = copy_and_replace_metadata(
        old_metadata=first_arr.metadata, new_shape=new_shape, new_chunks=new_chunks
    )

    return ManifestArray(chunkmanifest=stacked_manifest, metadata=new_metadata)


@implements(np.expand_dims)
def expand_dims(x: "ManifestArray", /, *, axis: int = 0) -> "ManifestArray":
    """Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by axis."""
    # this is just a special case of stacking
    return stack([x], axis=axis)


@implements(np.broadcast_to)
def broadcast_to(x: "ManifestArray", /, shape: tuple[int, ...]) -> "ManifestArray":
    """
    Broadcasts a ManifestArray to a specified shape, by either adjusting chunk keys or copying chunk manifest entries.
    """

    from .array import ManifestArray

    new_shape = shape

    # check its actually possible to broadcast to this new shape
    mutually_broadcastable_shape = np.broadcast_shapes(x.shape, new_shape)
    if mutually_broadcastable_shape != new_shape:
        # we're not trying to broadcast both shapes to a third shape
        raise ValueError(
            f"array of shape {x.shape} cannot be broadcast to shape {new_shape}"
        )

    # new chunk_shape is old chunk_shape with singleton dimensions prepended
    # (chunk shape can never change by more than adding length-1 axes because each chunk represents a fixed number of array elements)
    # broadcast_to only applies to regular chunk grids
    old_chunk_shape = x.chunk_grid.chunk_shape
    new_chunk_shape = _prepend_singleton_dimensions(
        old_chunk_shape, ndim=len(new_shape)
    )

    new_metadata = copy_and_replace_metadata(
        old_metadata=x.metadata,
        new_shape=list(new_shape),
        new_chunks=list(new_chunk_shape),
    )
    new_chunk_grid_shape = ChunkGrid.from_metadata(new_metadata).grid_shape

    # do broadcasting of entries in manifest
    broadcasted_manifest = _broadcast_manifest(x.manifest, shape=new_chunk_grid_shape)

    return ManifestArray(chunkmanifest=broadcasted_manifest, metadata=new_metadata)


def _concat_manifests(manifests: list[ChunkManifest], axis: int) -> ChunkManifest:
    """Concatenate manifests along an existing axis."""
    concatenated_paths = cast(
        np.ndarray[Any, np.dtypes.StringDType],
        np.concatenate([m._paths for m in manifests], axis=axis),
    )
    concatenated_offsets = np.concatenate([m._offsets for m in manifests], axis=axis)
    concatenated_lengths = np.concatenate([m._lengths for m in manifests], axis=axis)

    # merge inlined chunk dicts with index shifting along the concat axis
    concatenated_inlined: dict[tuple[int, ...], bytes] = {}
    grid_offset = 0
    for m in manifests:
        for key, data in m._inlined.items():
            shifted = list(key)
            shifted[axis] += grid_offset
            concatenated_inlined[tuple(shifted)] = data
        grid_offset += m._paths.shape[axis]

    return ChunkManifest.from_arrays(
        paths=concatenated_paths,
        offsets=concatenated_offsets,
        lengths=concatenated_lengths,
        validate_paths=False,
        inlined=concatenated_inlined if concatenated_inlined else None,
    )


def _stack_manifests(manifests: list[ChunkManifest], axis: int) -> ChunkManifest:
    """Stack manifests along a new axis."""
    stacked_paths = cast(
        np.ndarray[Any, np.dtypes.StringDType],
        np.stack([m._paths for m in manifests], axis=axis),
    )
    stacked_offsets = np.stack([m._offsets for m in manifests], axis=axis)
    stacked_lengths = np.stack([m._lengths for m in manifests], axis=axis)

    # merge inlined chunk dicts, inserting the new stacked axis
    stacked_inlined: dict[tuple[int, ...], bytes] = {}
    for i, m in enumerate(manifests):
        for key, data in m._inlined.items():
            shifted = list(key)
            shifted.insert(axis, i)
            stacked_inlined[tuple(shifted)] = data

    return ChunkManifest.from_arrays(
        paths=stacked_paths,
        offsets=stacked_offsets,
        lengths=stacked_lengths,
        validate_paths=False,
        inlined=stacked_inlined if stacked_inlined else None,
    )


def _broadcast_manifest(
    manifest: ChunkManifest, shape: tuple[int, ...]
) -> ChunkManifest:
    """Broadcast manifest to a new chunk grid shape."""
    broadcasted_paths = cast(
        np.ndarray[Any, np.dtypes.StringDType],
        np.broadcast_to(manifest._paths, shape=shape),
    )
    broadcasted_offsets = np.broadcast_to(manifest._offsets, shape=shape)
    broadcasted_lengths = np.broadcast_to(manifest._lengths, shape=shape)

    # broadcast inlined chunks: prepend singleton dims to each key, then replicate
    # the entry across every target position along any axis that was size 1 in the
    # source (matching np.broadcast_to semantics for the paths/offsets/lengths arrays).
    broadcasted_inlined: dict[tuple[int, ...], bytes] = {}
    if manifest._inlined:
        n_prepended = len(shape) - manifest._paths.ndim
        source_shape_padded = (1,) * n_prepended + manifest._paths.shape
        for key, data in manifest._inlined.items():
            padded_key = (0,) * n_prepended + key
            axis_ranges = [
                range(shape[i]) if source_shape_padded[i] == 1 else (padded_key[i],)
                for i in range(len(shape))
            ]
            for target_key in itertools.product(*axis_ranges):
                broadcasted_inlined[target_key] = data

    return ChunkManifest.from_arrays(
        paths=broadcasted_paths,
        offsets=broadcasted_offsets,
        lengths=broadcasted_lengths,
        validate_paths=False,
        inlined=broadcasted_inlined if broadcasted_inlined else None,
    )


def _prepend_singleton_dimensions(shape: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Prepend as many new length-1 axes to shape as necessary such that the result has ndim number of axes."""
    n_prepended_dims = ndim - len(shape)
    return tuple([1] * n_prepended_dims + list(shape))


# TODO broadcast_arrays, squeeze, permute_dims


@implements(np.full_like)
def full_like(
    x: "ManifestArray", /, fill_value: bool, *, dtype: np.dtype | None
) -> np.ndarray:
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.

    Returns a numpy array instead of a ManifestArray.

    Only implemented to get past some checks deep inside xarray, see https://github.com/zarr-developers/VirtualiZarr/issues/29.
    For creating a ManifestArray placeholder backed entirely by a fill_value, use
    :meth:`ManifestArray.fill_value_placeholder` instead.
    """
    return np.full(
        shape=x.shape,
        fill_value=fill_value,
        dtype=dtype if dtype is not None else x.dtype,
    )


@implements(np.isnan)
def isnan(x: "ManifestArray", /) -> np.ndarray:
    """
    Returns a numpy array of all False.

    Only implemented to get past some checks deep inside xarray, see https://github.com/zarr-developers/VirtualiZarr/issues/29.
    """
    return _isnan(x.shape)


def _isnan(shape: tuple):
    return np.full(shape=shape, fill_value=False, dtype=np.dtype(bool))
