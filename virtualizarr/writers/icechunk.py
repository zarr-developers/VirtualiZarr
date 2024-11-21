from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import zarr  # type: ignore[import-untyped]
from xarray import Dataset
from xarray.backends.zarr import encode_zarr_attr_value
from xarray.core.variable import Variable
from zarr import Array

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.utils import (
    check_combinable_zarr_arrays,
    check_compatible_encodings,
    check_same_ndims,
    check_same_shapes_except_on_concat_axis,
)
from virtualizarr.zarr import encode_dtype

if TYPE_CHECKING:
    from icechunk import IcechunkStore  # type: ignore[import-not-found]
    from zarr import Group  # type: ignore


VALID_URI_PREFIXES = {
    "s3://",
    # "gs://",  # https://github.com/earth-mover/icechunk/issues/265
    # "azure://",  # https://github.com/earth-mover/icechunk/issues/266
    # "r2://",
    # "cos://",
    # "minio://",
    "file:///",
}


def dataset_to_icechunk(
    ds: Dataset, store: "IcechunkStore", append_dim: Optional[str] = None
) -> None:
    """
    Write an xarray dataset whose variables wrap ManifestArrays to an Icechunk store.

    Currently requires all variables to be backed by ManifestArray objects.

    Parameters
    ----------
    ds: xr.Dataset
    store: IcechunkStore
    """
    try:
        from icechunk import IcechunkStore  # type: ignore[import-not-found]
        from zarr import Group  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'icechunk' and 'zarr' version 3 libraries are required to use this function"
        )

    if not isinstance(store, IcechunkStore):
        raise TypeError(f"expected type IcechunkStore, but got type {type(store)}")

    if not store.supports_writes:
        raise ValueError("supplied store does not support writes")

    # TODO only supports writing to the root group currently
    # TODO pass zarr_format kwarg?
    if store.mode.str == "a":
        if append_dim is None:
            raise ValueError(
                "append_dim must be provided when opening store in append mode"
            )
        root_group = Group.open(store=store, zarr_format=3)
    else:
        root_group = Group.from_store(store=store)

    # TODO this is Frozen, the API for setting attributes must be something else
    # root_group.attrs = ds.attrs
    # for k, v in ds.attrs.items():
    #     root_group.attrs[k] = encode_zarr_attr_value(v)

    return write_variables_to_icechunk_group(
        ds.variables,
        ds.attrs,
        store=store,
        group=root_group,
        append_dim=append_dim,
    )


def write_variables_to_icechunk_group(
    variables,
    attrs,
    store,
    group,
    append_dim: Optional[str] = None,
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
        store, zarr_format=3, consolidated=False, mode="a", append_dim=append_dim
    )

    # Then finish by writing the virtual variables to the same group
    for name, var in virtual_variables.items():
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
            append_dim=append_dim,
        )


def num_chunks(
    array,
    axis: int,
):
    return array.shape[axis] // array.chunks[axis]


def resize_array(
    group: "Group",
    name: str,
    var: Variable,
    append_axis: int,
) -> Array:
    existing_array = group[name]
    # This is the second time we check if the array is an instance of zarr.core.Array
    # but it's necessary to ensure .shape and .resize are available
    if not isinstance(existing_array, Array):
        raise ValueError("Expected existing array to be a zarr.core.Array")
    new_shape = list(existing_array.shape)
    new_shape[append_axis] += var.shape[append_axis]
    existing_array.resize(tuple(new_shape))
    return existing_array


def get_axis(
    dims: list[str],
    dim_name: Optional[str],
) -> int:
    if dim_name is None:
        raise ValueError("dim_name must be provided")
    return dims.index(dim_name)


def check_compatible_arrays(
    ma: ManifestArray, existing_array: zarr.core.array.Array, append_axis: int
):
    check_combinable_zarr_arrays([ma, existing_array])
    check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: Variable,
    append_dim: Optional[str] = None,
) -> None:
    """Write a single virtual variable into an icechunk store"""
    ma = cast(ManifestArray, var.data)
    zarray = ma.zarray
    mode = store.mode.str

    dims: list[str] = cast(list[str], list(var.dims))
    append_axis, existing_num_chunks, arr = None, None, None
    if append_dim and append_dim not in dims:
        raise ValueError(
            f"append_dim {append_dim} not found in variable dimensions {dims}"
        )
    if mode == "a":
        existing_array = group[name]
        if not isinstance(existing_array, Array):
            raise ValueError("Expected existing array to be a zarr.core.Array")
        append_axis = get_axis(dims, append_dim)

        # check if arrays can be concatenated
        check_compatible_arrays(ma, existing_array, append_axis)
        check_compatible_encodings(var.encoding, existing_array.attrs)

        # determine number of existing chunks along the append axis
        existing_num_chunks = num_chunks(
            array=group[name],
            axis=append_axis,
        )

        arr = group[name]

        # resize the array
        resize_array(
            arr,
            shape_to_append=var.shape,
            append_axis=append_axis,
        )
    else:
        # create array if it doesn't already exist
        arr = group.require_array(
            name=name,
            shape=zarray.shape,
            chunk_shape=zarray.chunks,
            dtype=encode_dtype(zarray.dtype),
            codecs=zarray._v3_codec_pipeline(),
            dimension_names=var.dims,
            fill_value=zarray.fill_value,
        )

        # TODO it would be nice if we could assign directly to the .attrs property
        for k, v in var.attrs.items():
            arr.attrs[k] = encode_zarr_attr_value(v)

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
    )


def generate_chunk_key(
    index: tuple[int, ...],
    append_axis: Optional[int] = None,
    existing_num_chunks: Optional[int] = None,
) -> str:
    if append_axis and append_axis >= len(index):
        raise ValueError(
            f"append_axis {append_axis} is greater than the number of indices {len(index)}"
        )
    return "/".join(
        str(ind + existing_num_chunks)
        if axis is append_axis and existing_num_chunks is not None
        else str(ind)
        for axis, ind in enumerate(index)
    )


def write_manifest_virtual_refs(
    store: "IcechunkStore",
    group: "Group",
    arr_name: str,
    manifest: ChunkManifest,
    append_axis: Optional[int] = None,
    existing_num_chunks: Optional[int] = None,
) -> None:
    """Write all the virtual references for one array manifest at once."""

    key_prefix = f"{group.name}{arr_name}"

    # loop over every reference in the ChunkManifest for that array
    # TODO inefficient: this should be replaced with something that sets all (new) references for the array at once
    # but Icechunk need to expose a suitable API first
    it = np.nditer(
        [manifest._paths, manifest._offsets, manifest._lengths],  # type: ignore[arg-type]
        flags=[
            "refs_ok",
            "multi_index",
            "c_index",
        ],
        op_flags=[["readonly"]] * 3,  # type: ignore
    )

    for path, offset, length in it:
        # it.multi_index will be an iterator of the chunk shape
        index = it.multi_index
        chunk_key = generate_chunk_key(index, append_axis, existing_num_chunks)

        # set each reference individually
        store.set_virtual_ref(
            # TODO it would be marginally neater if I could pass the group and name as separate args
            key=f"{key_prefix}/c/{chunk_key}",  # should be of form 'group/arr_name/c/0/1/2', where c stands for chunks
            location=as_file_uri(path.item()),
            offset=offset.item(),
            length=length.item(),
        )


def as_file_uri(path):
    # TODO a more robust solution to this requirement exists in https://github.com/zarr-developers/VirtualiZarr/pull/243
    if not any(path.startswith(prefix) for prefix in VALID_URI_PREFIXES) and path != "":
        # assume path is local
        return f"file://{path}"
    else:
        return path
