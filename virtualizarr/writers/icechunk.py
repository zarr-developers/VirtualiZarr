import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

import numpy as np
import xarray as xr
from xarray.backends.zarr import ZarrStore as XarrayZarrStore
from xarray.backends.zarr import encode_zarr_attr_value
from zarr import Array, Group, create_array, open_group
from zarr.core.buffer import default_buffer_prototype
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.errors import ContainsGroupError, GroupNotFoundError
from zarr.storage import MemoryStore, StorePath

from virtualizarr.codecs import extract_codecs, get_codecs
from virtualizarr.manifests import ChunkManifest, ManifestArray, ManifestGroup
from virtualizarr.manifests.manifest import INLINED_CHUNK_PATH
from virtualizarr.manifests.utils import (
    check_compatible_encodings,
    check_same_chunk_shapes,
    check_same_codecs,
    check_same_dtypes,
    check_same_ndims,
    check_same_shapes_except_axes,
    check_same_shapes_except_on_concat_axis,
)

if TYPE_CHECKING:
    from icechunk import (
        IcechunkStore,  # type: ignore[import-not-found]
        RepositoryConfig,  # type: ignore[import-not-found]
    )


ENCODING_KEYS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}

VALID_MODES = ("w", "w-", "a")


def _resolve_mode(
    mode: Optional[Literal["w", "w-", "a"]],
    append_dim: Optional[str] = None,
    region: object = None,
) -> Literal["w", "w-", "a", "r+"]:
    """Validate ``mode`` and resolve it to the effective zarr group-open mode."""
    if not isinstance(mode, (type(None), str)):
        raise TypeError(f"mode: expected type Optional[str], but got type {type(mode)}")

    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"mode: expected one of {VALID_MODES}, but got {mode!r}")

    if append_dim or region:
        if mode in ("w", "w-"):
            raise ValueError(
                f"mode {mode!r} cannot be used together with append_dim or region, "
                "which require opening an existing group"
            )
        # appending or writing to a region requires the group (and arrays) to already exist
        return "r+"

    return mode or "w-"


def _check_store_and_last_updated_at(
    store: "IcechunkStore", last_updated_at: Optional[datetime]
) -> None:
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "The 'icechunk' and 'zarr' version 3 libraries are required to use this function"
        ) from None

    if not isinstance(store, IcechunkStore):
        raise TypeError(
            f"store: expected type IcechunkStore, but got type {type(store)}"
        )

    if not isinstance(last_updated_at, (type(None), datetime)):
        raise TypeError(
            "last_updated_at: expected type Optional[datetime],"
            f" but got type {type(last_updated_at)}"
        )

    if store.read_only:
        raise ValueError("supplied store is read-only")


def virtual_dataset_to_icechunk(
    vds: xr.Dataset,
    store: "IcechunkStore",
    *,
    group: Optional[str] = None,
    mode: Optional[Literal["w", "w-", "a"]] = None,
    append_dim: Optional[str] = None,
    region: Optional[Literal["auto"] | Mapping[str, Literal["auto"] | slice]] = None,
    validate_containers: bool = True,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write an virtual xarray dataset to an Icechunk store.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    vds
        Dataset to write to an Icechunk store. Can contain both "virtual" variables (backed by ManifestArray objects) and "loadable" variables (backed by numpy arrays).
    store
        Store to write the dataset to, which must not be read-only.
    group
        Path to the group in which to store the dataset, defaulting to the root group.
    mode
        How to handle a pre-existing group at the target path:

        - ``"w-"``: create the group, raising a ``ContainsGroupError`` if it already exists.
        - ``"w"``: create the group, overwriting any existing contents at that path.
        - ``"a"``: open the group if it exists (keeping existing arrays), otherwise create it.
          An existing array of the same name must have the same metadata apart from
          attributes, otherwise a ``ValueError`` is raised.
        - ``None`` (default): equivalent to ``"w-"``, unless ``append_dim`` or ``region``
          is given, in which case the existing group is opened.

        ``mode="w"`` and ``mode="w-"`` are incompatible with ``append_dim`` and ``region``.
    append_dim
        Name of the dimension along which to append data. If provided, the dataset must
        have a dimension with this name.
    region
        Optional mapping from dimension names to either a) ``"auto"``, or b) integer
        slices, indicating the region of existing zarr array(s) in which to write
        this dataset's data.

        See ``xarray.Dataset.to_zarr`` documentation for details.
    validate_containers
        If ``True``, raise if any virtual chunks refer to locations that don't
        match any existing virtual chunk container set on this Icechunk repository.

        It is not generally recommended to set this to ``False``, because it can lead to
        confusing runtime results and errors when reading data back.
    last_updated_at
        The time at which the virtual dataset was last updated. When specified, if any
        of the virtual chunks written in this session are modified in storage after this
        time, icechunk will raise an error at runtime when trying to read the virtual
        chunk. When not specified, icechunk will not check for modifications to the
        virtual chunks at runtime.

    Raises
    ------
    ValueError
        If the store is read-only.
    """
    _check_store_and_last_updated_at(store, last_updated_at)

    if not isinstance(group, (type(None), str)):
        raise TypeError(
            f"group: expected type Optional[str], but got type {type(group)}"
        )

    open_mode = _resolve_mode(mode, append_dim=append_dim, region=region)

    if not isinstance(append_dim, (type(None), str)):
        raise TypeError(
            f"append_dim: expected type Optional[str], but got type {type(append_dim)}"
        )

    if not isinstance(region, (type(None), str, Mapping)):
        raise TypeError(
            "region: expected type Optional[Literal['auto'] | Mapping[str, Literal['auto'] | slice]],"
            f" but got type {type(last_updated_at)}"
        )

    if append_dim and append_dim not in vds.dims:
        raise ValueError(
            f"append_dim {append_dim!r} does not match any existing dataset dimensions"
        )

    store_path = StorePath(store, path=group or "")

    if validate_containers:
        validate_virtual_chunk_containers(store.session.config, [vds])

    group_object = open_group(
        store_path, mode=open_mode, zarr_format=3, use_consolidated=False
    )

    write_virtual_dataset_to_icechunk_group(
        vds=vds,
        store=store,
        group=group_object,
        append_dim=append_dim,
        region=region,
        last_updated_at=last_updated_at,
    )


def virtual_datatree_to_icechunk(
    vdt: xr.DataTree,
    store: "IcechunkStore",
    *,
    mode: Optional[Literal["w", "w-", "a"]] = None,
    write_inherited_coords: bool = False,
    validate_containers: bool = True,
    last_updated_at: datetime | None = None,
    **kwargs,
) -> None:
    """
    Write an xarray dataset to an Icechunk store.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    vdt
        DataTree to write to an Icechunk store. Can contain both "virtual" variables (backed by ManifestArray objects) and "loadable" variables (backed by numpy arrays).
    store
        Store to write the dataset to, which must not be read-only.
    mode
        How to handle pre-existing groups at the target paths:

        - ``"w-"`` or ``None`` (default): create each group, raising a
          ``ContainsGroupError`` if it already exists.
        - ``"w"``: create each group, overwriting any existing contents at that path.
        - ``"a"``: open each group if it exists (keeping existing arrays), otherwise create it.
          An existing array of the same name must have the same metadata apart from
          attributes, otherwise a ``ValueError`` is raised.
    write_inherited_coords
        If ``True``, replicate inherited coordinates on all descendant nodes of the
        tree. Otherwise, only write coordinates at the level at which they are
        originally defined. This saves disk space, but requires opening the
        full tree to load inherited coordinates.
    validate_containers
        If ``True``, raise if any virtual chunks refer to locations that don't
        match any existing virtual chunk container set on this Icechunk repository.

        It is not generally recommended to set this to ``False``, because it can lead to
        confusing runtime results and errors when reading data back.
    last_updated_at
        The time at which the virtual dataset was last updated. When specified, if any
        of the virtual chunks written in this session are modified in storage after this
        time, icechunk will raise an error at runtime when trying to read the virtual
        chunk. When not specified, icechunk will not check for modifications to the
        virtual chunks at runtime.
    **kwargs
        Additional keyword arguments to be passed to ``xarray.Dataset.vz.to_icechunk``.

    Raises
    ------
    ValueError
        If the store is read-only.
    """
    _check_store_and_last_updated_at(store, last_updated_at)

    open_mode = _resolve_mode(
        mode, append_dim=kwargs.get("append_dim"), region=kwargs.get("region")
    )

    def node_to_vds(node: xr.DataTree) -> xr.Dataset:
        tree = cast(xr.DataTree, node)  # subtree is typed as Unknown
        at_root = tree is vdt
        return tree.to_dataset(write_inherited_coords or at_root)

    def get_store_path(subtree, vdt) -> StorePath:
        at_root = subtree is vdt
        return StorePath(store, path="" if at_root else subtree.relative_to(vdt))

    # can't just use a dict because StorePath is not hashable
    paths_and_virtual_datasets = [
        (get_store_path(subtree, vdt), node_to_vds(subtree)) for subtree in vdt.subtree
    ]
    virtual_datasets = [pair[1] for pair in paths_and_virtual_datasets]

    if validate_containers:
        validate_virtual_chunk_containers(store.session.config, virtual_datasets)

    # TODO this serial loop could be slow writing lots of groups to high-latency store, see https://github.com/pydata/xarray/issues/9455
    for store_path, vds in paths_and_virtual_datasets:
        group = open_group(
            store_path, mode=open_mode, zarr_format=3, use_consolidated=False
        )

        write_virtual_dataset_to_icechunk_group(
            vds=vds,
            store=store,
            group=group,
            last_updated_at=last_updated_at,
            **kwargs,
        )


def manifest_group_to_icechunk(
    manifest_group: ManifestGroup,
    store: "IcechunkStore",
    *,
    group: Optional[str] = None,
    mode: Optional[Literal["w", "w-", "a"]] = None,
    validate_containers: bool = True,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write a ManifestGroup, and all its subgroups, to an Icechunk store without going via xarray.

    Each array and group is written with the Zarr metadata it holds (dimension names,
    attributes, codecs and fill value). This can write structures an xarray Dataset
    can't hold, such as arrays without dimension names, or sibling arrays that share a
    dimension name at different lengths.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    manifest_group
        Group to write, with all its subgroups.
    store
        Store to write to, which must not be read-only.
    group
        Path to the group in which to write ``manifest_group``, defaulting to the root group.
    mode
        How to handle pre-existing groups at the target paths:

        - ``"w-"`` or ``None`` (default): create each group, raising a
          ``ContainsGroupError`` if it already exists.
        - ``"w"``: create each group, overwriting any existing contents at that path.
        - ``"a"``: open each group if it exists (keeping existing arrays), otherwise create it.
          An existing array of the same name must have the same metadata apart from
          attributes, otherwise a ``ValueError`` is raised. Every array is checked
          before any is written, so on this error nothing is written. The new
          references are written over the existing ones.
    validate_containers
        If ``True``, raise if any virtual chunks refer to locations that don't
        match any existing virtual chunk container set on this Icechunk repository.

        It is not generally recommended to set this to ``False``, because it can lead to
        confusing runtime results and errors when reading data back.
    last_updated_at
        The time at which the virtual references were last updated. When specified, if
        any of the virtual chunks written in this session are modified in storage after
        this time, icechunk will raise an error at runtime when trying to read the
        virtual chunk. When not specified, icechunk will not check for modifications to
        the virtual chunks at runtime.

    Raises
    ------
    ValueError
        If the store is read-only, ``mode`` is invalid, a virtual chunk refers to a
        location without a virtual chunk container (when ``validate_containers`` is set),
        or ``mode="a"`` would write over an existing array with different metadata.
    TypeError
        If an argument has the wrong type.
    zarr.errors.ContainsGroupError
        If a group already exists and ``mode`` is ``"w-"`` or ``None``, or if
        ``mode="a"`` and a group exists where an array would be written.
    """
    _check_store_and_last_updated_at(store, last_updated_at)

    if not isinstance(group, (type(None), str)):
        raise TypeError(
            f"group: expected type Optional[str], but got type {type(group)}"
        )

    open_mode = _resolve_mode(mode)

    paths_and_groups = list(_walk_manifest_group(manifest_group, path=group or ""))

    if validate_containers:
        _validate_manifest_arrays_have_containers(
            store.session.config,
            [arr for _, mgroup in paths_and_groups for arr in mgroup.arrays.values()],
        )

    if open_mode == "a":
        # under "w-" nothing needs checking: zarr creates parent groups, so any existing
        # group in the tree means the top one exists too, and opening it raises first
        _check_manifest_groups_can_be_appended(store, paths_and_groups)

    # parents come before children, so mode="w" on a parent cannot erase a child already written
    for path, mgroup in paths_and_groups:
        zarr_group = open_group(
            StorePath(store, path=path),
            mode=open_mode,
            zarr_format=3,
            use_consolidated=False,
        )
        for name, marr in mgroup.arrays.items():
            _write_manifest_array_to_icechunk(
                store=store,
                group=zarr_group,
                name=name,
                marr=marr,
                last_updated_at=last_updated_at,
            )
        zarr_group.update_attributes(mgroup.metadata.attributes)


def _walk_manifest_group(
    manifest_group: ManifestGroup, path: str
) -> Iterator[tuple[str, ManifestGroup]]:
    """Yield (path, group) for this group and every subgroup below it, parents first."""
    yield path, manifest_group
    for name, subgroup in manifest_group.groups.items():
        yield from _walk_manifest_group(subgroup, f"{path}/{name}" if path else name)


def _check_manifest_groups_can_be_appended(
    store: "IcechunkStore",
    paths_and_groups: Sequence[tuple[str, ManifestGroup]],
) -> None:
    """
    Raise before anything is written if any array in the tree would fail to write under ``mode="a"``.

    Checking up front keeps a failure from leaving part of the tree in the session.
    """
    for path, mgroup in paths_and_groups:
        try:
            zarr_group = open_group(
                StorePath(store, path=path),
                mode="r",
                zarr_format=3,
                use_consolidated=False,
            )
        except GroupNotFoundError:
            continue
        for name, marr in mgroup.arrays.items():
            existing = zarr_group.get(name)
            if isinstance(existing, Group):
                raise ContainsGroupError(store, f"{path}/{name}" if path else name)
            if isinstance(existing, Array):
                _check_existing_array_matches(
                    existing,
                    _virtual_array_kwargs(marr.metadata, marr.metadata.dimension_names),
                )


def _write_manifest_array_to_icechunk(
    store: "IcechunkStore",
    group: Group,
    name: str,
    marr: ManifestArray,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write one ManifestArray, with its own Zarr metadata, into an existing zarr group of an icechunk store.

    Any existing array of this name must already have been checked with
    ``_check_existing_array_matches``.
    """
    metadata = marr.metadata
    arr = group.require_array(
        name=name, **_virtual_array_kwargs(metadata, metadata.dimension_names)
    )
    arr.update_attributes(metadata.attributes)

    write_manifest_to_icechunk(
        store=store,
        group=group,
        arr_name=name,
        manifest=marr.manifest,
        chunk_index_offsets=(0,) * marr.ndim,
        last_updated_at=last_updated_at,
    )


# TODO ideally I would be able to just call some Icechunk API to do this (see https://github.com/earth-mover/icechunk/issues/1167)
def validate_virtual_chunk_containers(
    config: "RepositoryConfig", virtual_datasets: Iterable[xr.Dataset]
) -> None:
    """Check that all virtual refs have corresponding virtual chunk containers, before writing any of the refs."""

    manifestarrays = [
        var.data
        for dataset in virtual_datasets
        for var in dataset.variables.values()
        if isinstance(var.data, ManifestArray)
    ]
    _validate_manifest_arrays_have_containers(config, manifestarrays)


def _validate_manifest_arrays_have_containers(
    config: "RepositoryConfig", manifestarrays: Sequence[ManifestArray]
) -> None:
    """Raise if any ref in these ManifestArrays has no matching virtual chunk container."""

    # get the prefixes of all virtual chunk containers
    if config.virtual_chunk_containers is None:
        # TODO for some reason Icechunk returns None instead of an empty dict if there are zero containers (see https://github.com/earth-mover/icechunk/issues/1168)
        supported_prefixes = set()
    else:
        supported_prefixes = set(config.virtual_chunk_containers.keys())

    # fastpath for common case that no virtual chunk containers have been set
    if manifestarrays and not supported_prefixes:
        raise ValueError("No Virtual Chunk Containers set")

    # check all refs against existing virtual chunk containers
    # passing a tuple to str.startswith runs the loop over prefixes in C
    supported_prefixes_tuple = tuple(supported_prefixes)
    for marr in manifestarrays:
        # TODO this loop over every virtual reference is likely inefficient in python,
        # is there a way to push this down to Icechunk? (see https://github.com/earth-mover/icechunk/issues/1167)
        for ref in marr.manifest.iter_nonempty_paths():
            validate_single_ref(ref, supported_prefixes_tuple)


def validate_single_ref(ref: str, supported_prefixes: tuple[str, ...]) -> None:
    if not ref.startswith(supported_prefixes):
        raise ValueError(
            f"No Virtual Chunk Container set which supports prefix of path {ref}"
        )


def write_virtual_dataset_to_icechunk_group(
    vds: xr.Dataset,
    store: "IcechunkStore",
    group: Group,
    append_dim: Optional[str] = None,
    region: Optional[Literal["auto"] | Mapping[str, Literal["auto"] | slice]] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    if region is not None:
        vds, region = validate_and_autodetect_region(group, vds, region)

    virtual_variables = {
        name: var
        for name, var in vds.variables.items()
        if isinstance(var.data, ManifestArray)
    }

    loadable_variables = {
        name: var
        for name, var in vds.variables.items()
        if name not in virtual_variables
    }

    # First write all the non-virtual variables
    if loadable_variables:
        mode: Literal["a", "r+"] | None
        if append_dim is None and region is None:
            mode = "a"
        else:
            # let xarray set it automatically, 'a' for append_dim and 'r+' for region
            mode = None
        loadable_ds = xr.Dataset(loadable_variables)
        loadable_ds.to_zarr(  # type: ignore[call-overload]
            store,
            group=group.name,
            zarr_format=3,
            consolidated=False,
            mode=mode,
            append_dim=append_dim,
            region=region,
        )

    # Then write the virtual variables to the same group
    for name, var in virtual_variables.items():
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,  # type: ignore[arg-type]
            var=var,
            append_dim=append_dim,
            region=region,
            last_updated_at=last_updated_at,
        )

    # finish by writing group-level attributes
    # note: group attributes must be set after writing individual variables else it gets overwritten
    update_attributes(group, vds.attrs, coords=vds.coords)


def update_attributes(
    zarr_node: Array | Group, attrs: dict, coords=None, encoding=None
):
    """Update metadata attributes of one Zarr node (array or group), to match how xarray does it."""

    zarr_node.update_attributes(
        {k: encode_zarr_attr_value(v) for k, v in attrs.items()}
    )

    # preserve info telling xarray which variables are coordinates upon re-opening
    if isinstance(zarr_node, Group) and coords:
        zarr_node.update_attributes(
            {"coordinates": " ".join(list(coords))},
        )

    # preserve variable-level encoding
    if isinstance(zarr_node, Array) and encoding:
        for k, v in encoding.items():
            if k in ENCODING_KEYS:
                zarr_node.attrs[k] = encode_zarr_attr_value(v)


def validate_and_autodetect_region(
    group: Group,
    vds: xr.Dataset,
    region: Literal["auto"] | Mapping[str, Literal["auto"] | slice],
) -> tuple[xr.Dataset, dict[str, slice]]:
    """
    Convert regions like `"auto"` and `{"dim": "auto"}` into concrete `dict[str, slice]`
    in a way which is maximally compatible with xarray's `to_zarr`.
    The method itself is quite complicated, we call into xarray
    so we do not have to reimplement the logic.

    This function uses xarray's internal private functions and can break at any time.
    """
    xarray_store = XarrayZarrStore(
        zarr_group=group,
        append_dim=None,
        write_region=region,
        consolidate_on_close=False,
        close_store_on_close=False,
        mode="r+",
    )
    vds = xarray_store._validate_and_autodetect_region(vds)
    xarray_store.close()
    return vds, cast(dict[str, slice], xarray_store._write_region)


def num_chunks(
    array,
    axis: int,
) -> int:
    return array.shape[axis] // array.chunks[axis]


def resize_array(
    arr: "Array",
    manifest_array: "ManifestArray",
    append_axis: int,
) -> None:
    new_shape = list(arr.shape)
    new_shape[append_axis] += manifest_array.shape[append_axis]
    arr.resize(tuple(new_shape))


def get_axis(
    dims: list[str],
    dim_name: Optional[str],
) -> int:
    if dim_name is None:
        raise ValueError("dim_name must be provided")
    return dims.index(dim_name)


def check_compatible_arrays(
    ma: "ManifestArray",
    existing_array: "Array",
    append_axis: int | None,
    except_axes: list[int] | None = None,
):
    arrays: List[Union[ManifestArray, Array]] = [ma, existing_array]
    check_same_dtypes([arr.dtype for arr in arrays])
    check_same_codecs([get_codecs(arr) for arr in arrays])
    check_same_chunk_shapes([arr.metadata.chunks for arr in arrays])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    if append_axis is not None:
        check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)
    if except_axes is not None:
        check_same_shapes_except_axes(arr_shapes, except_axes)


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: xr.Variable,
    append_dim: Optional[str] = None,
    region: Optional[Mapping[str, slice]] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """Write a single virtual variable into an icechunk store"""

    ma = cast(ManifestArray, var.data)
    metadata = ma.metadata

    dims: list[str] = cast(list[str], list(var.dims))
    chunk_offsets: list[int]
    existing_num_chunks = 0
    if append_dim and append_dim in dims:
        # TODO: MRP - zarr, or icechunk zarr, array assignment to a variable doesn't work to point to the same object
        # for example, if you resize an array, it resizes the array but not the bound variable.
        if not isinstance(group[name], Array):
            raise ValueError("Expected existing array to be a zarr.core.Array")
        append_axis = get_axis(dims, append_dim)

        # check if arrays can be concatenated
        check_compatible_arrays(ma, group[name], append_axis)  # type: ignore[arg-type]
        check_compatible_encodings(var.encoding, group[name].attrs)

        # determine number of existing chunks along the append axis
        existing_num_chunks = num_chunks(
            array=group[name],
            axis=append_axis,
        )
        chunk_offsets = [
            existing_num_chunks if dim == append_dim else 0 for dim in dims
        ]

        # resize the array
        resize_array(
            group[name],  # type: ignore[arg-type]
            manifest_array=ma,
            append_axis=append_axis,
        )
    elif region is not None:
        existing_array = group[name]
        if not isinstance(existing_array, Array):
            raise ValueError(
                f"Expected {name!r} to be a zarr.core.Array, got {type(existing_array)}"
            )
        check_compatible_arrays(
            ma,
            existing_array,
            append_axis=None,
            except_axes=[
                get_axis(dims, region_dim)
                for region_dim in region.keys()
                if region_dim in dims
            ],
        )
        check_compatible_encodings(var.encoding, existing_array.attrs)

        chunk_offsets = []
        for dim, chunk_size in zip(dims, existing_array.chunks):
            if dim not in region:
                chunk_offsets.append(0)
                continue
            dim_region = region[dim]
            start = dim_region.start if dim_region.start is not None else 0
            if start % chunk_size != 0:
                raise ValueError(
                    f"Trying to write variable {name!r} to region "
                    + f"{region!r} in dimension {dim!r}, but it is not aligned to whole "
                    + f"chunks of size {chunk_size!r}"
                )
            chunk_offsets.append(start // chunk_size)
    else:
        chunk_offsets = [0 for _ in dims]
        arr = _require_virtual_array(group, name, metadata, dimension_names=dims)

        update_attributes(arr, var.attrs, encoding=var.encoding)

    write_manifest_to_icechunk(
        store=store,
        group=group,
        arr_name=name,
        manifest=ma.manifest,
        chunk_index_offsets=tuple(chunk_offsets),
        last_updated_at=last_updated_at,
    )


def _virtual_array_kwargs(
    metadata: ArrayV3Metadata, dimension_names: Iterable[str | None] | None
) -> dict:
    """
    Arguments to zarr's ``create_array`` for the array one ManifestArray's refs will be written into.

    The array takes its shape, chunks, shards, data type, codecs and fill value from
    ``metadata``. Attributes are left to the caller. The chunk key encoding is not
    carried over: a ManifestArray's is internal to the manifest, and chunks are always
    written at the default ``/``-separated keys.
    """
    filters, serializer, compressors = extract_codecs(metadata.inner_codecs)
    return dict(
        shape=metadata.shape,
        chunks=metadata.chunks,
        shards=metadata.shards,
        dtype=metadata.data_type.to_native_dtype(),
        filters=filters,
        compressors=compressors,
        serializer=serializer,
        dimension_names=dimension_names,
        fill_value=metadata.fill_value,
    )


def _check_existing_array_matches(existing: Array, array_kwargs: dict) -> None:
    """
    Raise if an existing array has different metadata, apart from attributes, than ``array_kwargs`` would create.

    Raises
    ------
    ValueError
        If any metadata differs. zarr's ``require_array`` would keep the stored
        metadata, so the new refs would be decoded with the wrong codecs or read under
        the wrong dimension names.
    """
    # build the requested metadata through zarr so it is normalized the same way as the stored metadata
    requested = create_array(MemoryStore(), **array_kwargs).metadata.to_dict()
    stored = existing.metadata.to_dict()
    differing = sorted(
        key
        for key in requested.keys() | stored.keys()
        if key != "attributes" and requested.get(key) != stored.get(key)
    )
    if differing:
        raise ValueError(
            f"Array {existing.path!r} already exists with different "
            f"{', '.join(differing)}. Writing new references under the stored metadata "
            "would decode them incorrectly; use mode='w' to replace the group instead."
        )


def _require_virtual_array(
    group: Group,
    name: str,
    metadata: ArrayV3Metadata,
    dimension_names: Iterable[str | None] | None,
) -> Array:
    """
    Create or open the zarr array that one ManifestArray's refs will be written into.

    Raises
    ------
    ValueError
        If an array of this name already exists with different metadata.
    """
    array_kwargs = _virtual_array_kwargs(metadata, dimension_names)
    existing = group.get(name)
    if isinstance(existing, Array):
        _check_existing_array_matches(existing, array_kwargs)
    return group.require_array(name=name, **array_kwargs)


def write_manifest_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    arr_name: str,
    manifest: ChunkManifest,
    chunk_index_offsets: tuple[int, ...],
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write all the chunks (virtual and/or inlined) for one array manifest at once.

    Virtual chunks are written as virtual chunks, and inlined chunks are written as native
    (which Icechunk may then choose to inline in its manifests).
    """

    if group.name == "/":
        key_prefix = arr_name
    else:
        key_prefix = f"{group.name}/{arr_name}"

    if last_updated_at is None:
        # Icechunk rounds timestamps to the nearest second, but filesystems have higher precision,
        # so we need to add a buffer, so that if you immediately read data back from this icechunk store,
        # and the referenced data was literally just created (<1s ago),
        # you don't get an IcechunkError warning you that your referenced chunk has changed.
        # In practice this should only really come up in synthetic examples, e.g. tests and docs.
        last_updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)

    paths_flat = manifest._paths.flatten()

    if manifest._inlined:
        # Write inlined chunks first, then erase them from the paths array so the
        # virtual-refs write below doesn't see the INLINED_CHUNK_PATH sentinel
        # (which Icechunk's `.set_virtual_refs_arr` would reject as a malformed URL).
        # Use of zarr's `sync` here is to avoid a serial high-latency loop over chunks.
        # Would prefer if zarr-python had a public API for setting many chunks at once concurrently.
        sync(
            write_inlined_chunks_as_native(
                store=store,
                key_prefix=key_prefix,
                inlined=manifest._inlined,
                chunk_index_offsets=chunk_index_offsets,
            )
        )
        virtual_paths = np.where(paths_flat == INLINED_CHUNK_PATH, "", paths_flat)
    else:
        virtual_paths = paths_flat

    # Cheap numpy-level check so we can skip the .tolist() allocation and the
    # Python->Rust call entirely when no position holds a real virtual ref
    # (e.g. an all-inlined or all-missing manifest).
    if (virtual_paths != "").any():
        # Pass flat per-chunk arrays (or a list) to Rust in one call, avoiding Python-side
        # per-chunk dict construction. Empty paths are skipped on the Rust side.
        store.set_virtual_refs_arr(
            array_path=key_prefix,
            chunk_grid_shape=manifest.shape_chunk_grid,
            locations=virtual_paths.tolist(),
            offsets=manifest._offsets.flatten(),
            lengths=manifest._lengths.flatten(),
            validate_containers=False,
            arr_offset=chunk_index_offsets if any(chunk_index_offsets) else None,
            checksum=last_updated_at,
        )


async def write_inlined_chunks_as_native(
    store: "IcechunkStore",
    key_prefix: str,
    inlined: Mapping[tuple[int, ...], bytes],
    chunk_index_offsets: tuple[int, ...],
) -> None:
    """Write each inlined chunk as a native chunk at its zarr chunk key.

    Icechunk's ``Key::parse`` only accepts the standard zarr v3 ``/``-separated
    ``c/i0/i1/...`` chunk-key form; the manifest's stored chunk_key_encoding
    uses ``.`` (the ManifestStore's internal convention), so we encode with a
    slash-separated encoding here regardless.
    """
    encoding = DefaultChunkKeyEncoding(separator="/")
    prototype = default_buffer_prototype()
    has_offset = any(chunk_index_offsets)
    coros = []
    for chunk_idx, data in inlined.items():
        shifted_idx = (
            tuple(c + o for c, o in zip(chunk_idx, chunk_index_offsets))
            if has_offset
            else chunk_idx
        )
        encoded_chunk_key = encoding.encode_chunk_key(shifted_idx)
        coros.append(
            store.set(
                f"{key_prefix}/{encoded_chunk_key}",
                prototype.buffer.from_bytes(data),
            )
        )
    await asyncio.gather(*coros)
