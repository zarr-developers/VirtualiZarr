from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union, cast

import numpy as np
from xarray import Dataset
from xarray.backends.zarr import encode_zarr_attr_value
from xarray.core.variable import Variable

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
from virtualizarr.zarr import encode_dtype

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]
    from zarr import Array, Group  # type: ignore


def dataset_to_icechunk(
    ds: Dataset,
    store: "IcechunkStore",
    *,
    group: Optional[str] = None,
    append_dim: Optional[str] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """
    Write an xarray dataset to an Icechunk store.

    Both `icechunk` and `zarr` (v3) must be installed.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to write to an Icechunk store. All variables must be backed by
        ManifestArray objects.
    store: IcechunkStore
        Store to write the dataset to, which must not be read-only.
    group: Optional[str]
        Path to the group in which to store the dataset, defaulting to the root group.
    append_dim: Optional[str]
        Name of the dimension along which to append data. If provided, the dataset must
        have a dimension with this name.
    last_updated_at: Optional[datetime]
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

    if append_dim and append_dim not in ds.dims:
        raise ValueError(
            f"append_dim {append_dim!r} does not match any existing dataset dimensions"
        )

    store_path = StorePath(store, path=group or "")

    if append_dim:
        group_object = Group.open(store=store_path, zarr_format=3)
    else:
        group_object = Group.from_store(store=store_path, zarr_format=3)

    group_object.update_attributes(
        {k: encode_zarr_attr_value(v) for k, v in ds.attrs.items()}
    )

    return write_variables_to_icechunk_group(
        ds.variables,
        ds.attrs,
        store=store,
        group=group_object,
        append_dim=append_dim,
        last_updated_at=last_updated_at,
    )


def write_variables_to_icechunk_group(
    variables,
    attrs,
    store,
    group: "Group",
    append_dim: Optional[str] = None,
    last_updated_at: Optional[datetime] = None,
):
    virtual_variables = {
        name: var
        for name, var in variables.items()
        if isinstance(var.data, ManifestArray)
    }

    loadable_variables = {
        name: var for name, var in variables.items() if name not in virtual_variables
    }

    # First write all the non-virtual variables
    # NOTE: We set the attributes of the group before writing the dataset because the dataset
    # will overwrite the root group's attributes with the dataset's attributes. We take advantage
    # of xarrays zarr integration to ignore having to format the attributes ourselves.
    ds = Dataset(loadable_variables, attrs=attrs)
    ds.to_zarr(
        store,
        group=group.name,
        zarr_format=3,
        consolidated=False,
        mode="a",
        append_dim=append_dim,
    )

    # Then finish by writing the virtual variables to the same group
    for name, var in virtual_variables.items():
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
            append_dim=append_dim,
            last_updated_at=last_updated_at,
        )


def num_chunks(
    array,
    axis: int,
):
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
    check_same_codecs([get_codecs(arr, normalize_to_zarr_v3=True) for arr in arrays])
    check_same_chunk_shapes([arr.chunks for arr in arrays])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: Variable,
    append_dim: Optional[str] = None,
    last_updated_at: Optional[datetime] = None,
) -> None:
    """Write a single virtual variable into an icechunk store"""
    from zarr import Array

    ma = cast(ManifestArray, var.data)
    zarray = ma.zarray

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

        # Get the codecs and convert them to zarr v3 format
        codecs = zarray._v3_codecs()

        # create array if it doesn't already exist
        arr = group.require_array(
            name=name,
            shape=zarray.shape,
            chunks=zarray.chunks,
            dtype=encode_dtype(zarray.dtype),
            compressors=codecs.compressors,
            serializer=codecs.serializer,
            filters=codecs.filters,
            dimension_names=var.dims,
            fill_value=zarray.fill_value,
        )

        arr.update_attributes(
            {k: encode_zarr_attr_value(v) for k, v in var.attrs.items()}
        )

        _encoding_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}
        for k, v in var.encoding.items():
            if k in _encoding_keys:
                arr.attrs[k] = encode_zarr_attr_value(v)

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

    virtual_chunk_spec_list = [
        VirtualChunkSpec(
            index=generate_chunk_key(it.multi_index, append_axis, existing_num_chunks),
            location=path.item(),
            offset=offset.item(),
            length=length.item(),
            last_updated_at_checksum=last_updated_at,
        )
        for path, offset, length in it
    ]

    store.set_virtual_refs(array_path=key_prefix, chunks=virtual_chunk_spec_list)
