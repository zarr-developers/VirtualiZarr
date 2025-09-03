from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable, List, Optional, Union, cast

import numpy as np
import xarray as xr
from xarray.backends.zarr import encode_zarr_attr_value
from zarr import Array, Group

from virtualizarr.codecs import get_codecs
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import (
    check_compatible_encodings,
    check_same_chunk_shapes,
    check_same_codecs,
    check_same_dtypes,
    check_same_ndims,
    check_same_shapes_except_on_concat_axis,
)

if TYPE_CHECKING:
    from icechunk import (
        IcechunkStore,  # type: ignore[import-not-found]
        RepositoryConfig,  # type: ignore[import-not-found]
    )


ENCODING_KEYS = {"_FillValue", "missing_value", "scale_factor", "add_offset"}


def virtual_dataset_to_icechunk(
    vds: xr.Dataset,
    store: "IcechunkStore",
    *,
    group: Optional[str] = None,
    append_dim: Optional[str] = None,
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
    append_dim
        Name of the dimension along which to append data. If provided, the dataset must
        have a dimension with this name.
    validate_containers
        If ``True``, raise if any virtual chunks have a refer to locations that don't
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
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
        from zarr import Group  # type: ignore[import-untyped]
        from zarr.storage import StorePath  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'icechunk' and 'zarr' version 3 libraries are required to use this function"
        ) from None

    if not isinstance(store, IcechunkStore):
        raise TypeError(
            f"store: expected type IcechunkStore, but got type {type(store)}"
        )

    if not isinstance(group, (type(None), str)):
        raise TypeError(
            f"group: expected type Optional[str], but got type {type(group)}"
        )

    if not isinstance(append_dim, (type(None), str)):
        raise TypeError(
            f"append_dim: expected type Optional[str], but got type {type(append_dim)}"
        )

    if not isinstance(last_updated_at, (type(None), datetime)):
        raise TypeError(
            "last_updated_at: expected type Optional[datetime],"
            f" but got type {type(last_updated_at)}"
        )

    if store.read_only:
        raise ValueError("supplied store is read-only")

    if append_dim and append_dim not in vds.dims:
        raise ValueError(
            f"append_dim {append_dim!r} does not match any existing dataset dimensions"
        )

    store_path = StorePath(store, path=group or "")

    if validate_containers:
        validate_virtual_chunk_containers(store.session.config, [vds])

    if append_dim:
        group_object = Group.open(store=store_path, zarr_format=3)
    else:
        # create the group if it doesn't already exist
        group_object = Group.from_store(store=store_path, zarr_format=3)

    write_virtual_dataset_to_icechunk_group(
        vds=vds,
        store=store,
        group=group_object,
        append_dim=append_dim,
        last_updated_at=last_updated_at,
    )


def virtual_datatree_to_icechunk(
    vdt: xr.DataTree,
    store: "IcechunkStore",
    *,
    write_inherited_coords: bool = False,
    validate_containers: bool = True,
    last_updated_at: datetime | None = None,
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
    write_inherited_coords
        If ``True``, replicate inherited coordinates on all descendant nodes of the
        tree. Otherwise, only write coordinates at the level at which they are
        originally defined. This saves disk space, but requires opening the
        full tree to load inherited coordinates.
    validate_containers
        If ``True``, raise if any virtual chunks have a refer to locations that don't
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
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
        from zarr import Group  # type: ignore[import-untyped]
        from zarr.storage import StorePath  # type: ignore[import-untyped]
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
            "last_updated_at: expected type datetime,"
            f" but got type {type(last_updated_at)}"
        )

    if store.read_only:
        raise ValueError("supplied store is read-only")

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
        group = Group.from_store(store=store_path, zarr_format=3)

        write_virtual_dataset_to_icechunk_group(
            vds=vds,
            store=store,
            group=group,
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
    for marr in manifestarrays:
        # TODO this loop over every virtual reference is likely inefficient in python,
        # is there a way to push this down to Icechunk? (see https://github.com/earth-mover/icechunk/issues/1167)
        for ref in marr.manifest._paths.flat:
            if ref:
                validate_single_ref(ref, supported_prefixes)


def validate_single_ref(ref: str, supported_prefixes: set[str]) -> None:
    if not any(ref.startswith(prefix) for prefix in supported_prefixes):
        raise ValueError(
            f"No Virtual Chunk Container set which supports prefix of path {ref}"
        )


def write_virtual_dataset_to_icechunk_group(
    vds: xr.Dataset,
    store: "IcechunkStore",
    group: Group,
    append_dim: Optional[str] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
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
        loadable_ds = xr.Dataset(loadable_variables)
        loadable_ds.to_zarr(  # type: ignore[call-overload]
            store,
            group=group.name,
            zarr_format=3,
            consolidated=False,
            mode="a",
            append_dim=append_dim,
        )

    # Then write the virtual variables to the same group
    for name, var in virtual_variables.items():
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,  # type: ignore[arg-type]
            var=var,
            append_dim=append_dim,
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
    ma: "ManifestArray", existing_array: "Array", append_axis: int
):
    arrays: List[Union[ManifestArray, Array]] = [ma, existing_array]
    check_same_dtypes([arr.dtype for arr in arrays])
    check_same_codecs([get_codecs(arr) for arr in arrays])
    check_same_chunk_shapes([arr.chunks for arr in arrays])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: xr.Variable,
    append_dim: Optional[str] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """Write a single virtual variable into an icechunk store"""
    from zarr import Array

    from virtualizarr.codecs import extract_codecs

    ma = cast(ManifestArray, var.data)
    metadata = ma.metadata

    dims: list[str] = cast(list[str], list(var.dims))
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

        # resize the array
        resize_array(
            group[name],  # type: ignore[arg-type]
            manifest_array=ma,
            append_axis=append_axis,
        )
    else:
        append_axis = None
        # TODO: Should codecs be an argument to zarr's AsyncrGroup.create_array?
        filters, serializer, compressors = extract_codecs(metadata.codecs)
        arr = group.require_array(
            name=name,
            shape=metadata.shape,
            chunks=metadata.chunks,
            dtype=metadata.data_type,
            filters=filters,
            compressors=compressors,
            serializer=serializer,
            dimension_names=var.dims,
            fill_value=metadata.fill_value,
        )

        update_attributes(arr, var.attrs, encoding=var.encoding)

    write_manifest_virtual_refs(
        store=store,
        group=group,
        arr_name=name,
        manifest=ma.manifest,
        append_axis=append_axis,
        existing_num_chunks=existing_num_chunks,
        last_updated_at=last_updated_at,
    )


def generate_chunk_key(
    index: tuple[int, ...],
    append_axis: Optional[int] = None,
    existing_num_chunks: Optional[int] = None,
) -> list[int]:
    if append_axis and append_axis >= len(index):
        raise ValueError(
            f"append_axis {append_axis} is greater than the number of indices {len(index)}"
        )

    return [
        ind + existing_num_chunks
        if axis is append_axis and existing_num_chunks is not None
        else ind
        for axis, ind in enumerate(index)
    ]


def write_manifest_virtual_refs(
    store: "IcechunkStore",
    group: "Group",
    arr_name: str,
    manifest: ChunkManifest,
    append_axis: Optional[int] = None,
    existing_num_chunks: Optional[int] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """Write all the virtual references for one array manifest at once."""
    from icechunk import VirtualChunkSpec

    if group.name == "/":
        key_prefix = arr_name
    else:
        key_prefix = f"{group.name}/{arr_name}"

    # loop over every reference in the ChunkManifest for that array
    # TODO inefficient: this should be replaced with something that sets all (new) references for the array at once
    # but Icechunk need to expose a suitable API first
    # See https://github.com/earth-mover/icechunk/issues/401 for performance benchmark

    it = np.nditer(
        [manifest._paths, manifest._offsets, manifest._lengths],  # type: ignore[arg-type]
        flags=[
            "refs_ok",
            "multi_index",
            "c_index",
        ],
        op_flags=[["readonly"]] * 3,  # type: ignore
    )

    if last_updated_at is None:
        # Icechunk rounds timestamps to the nearest second, but filesystems have higher precision,
        # so we need to add a buffer, so that if you immediately read data back from this icechunk store,
        # and the referenced data was literally just created (<1s ago),
        # you don't get an IcechunkError warning you that your referenced chunk has changed.
        # In practice this should only really come up in synthetic examples, e.g. tests and docs.
        last_updated_at = datetime.now(timezone.utc) + timedelta(seconds=1)

    virtual_chunk_spec_list = [
        VirtualChunkSpec(
            index=generate_chunk_key(it.multi_index, append_axis, existing_num_chunks),
            location=path.item(),
            offset=offset.item(),
            length=length.item(),
            last_updated_at_checksum=last_updated_at,
        )
        for path, offset, length in it
        if path
    ]

    store.set_virtual_refs(
        array_path=key_prefix,
        chunks=virtual_chunk_spec_list,
        validate_containers=False,  # we already validated these before setting any refs
    )
