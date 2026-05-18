from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import numpy as np

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

    new_shape: list[int] = []
    new_chunks: list[int] = []
    chunk_grid_selectors: list[int | slice] = []
    kept_axes: list[int] = []
    for axis, (axis_length, chunk_size, indexer_1d) in enumerate(
        zip(marr.shape, marr.chunks, narrowed_indexers, strict=True)
    ):
        chunk_grid_selector, new_axis_length = _compute_chunk_aligned_selection_1d(
            indexer_1d, axis_length=axis_length, chunk_size=chunk_size
        )
        chunk_grid_selectors.append(chunk_grid_selector)
        # int selectors drop the axis from the output array
        if not isinstance(indexer_1d, int):
            new_shape.append(new_axis_length)
            new_chunks.append(chunk_size)
            kept_axes.append(axis)

    chunk_grid_selectors_tuple = tuple(chunk_grid_selectors)

    # short-circuit if every axis selects the whole chunk grid via a slice (a no-op)
    if all(
        isinstance(cgs, slice) and cgs == slice(0, dim, 1)
        for cgs, dim in zip(chunk_grid_selectors_tuple, marr.manifest.shape_chunk_grid)
    ):
        return marr

    new_manifest = _subset_manifest(marr.manifest, chunk_grid_selectors_tuple)
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
