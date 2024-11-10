from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import zarr
from xarray import Dataset
from xarray.backends.zarr import encode_zarr_attr_value
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests import array_api as manifest_api
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


def write_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: Variable,
    append_dim: Optional[str] = None,
) -> None:
    """Write a single (possibly virtual) variable into an icechunk store"""
    if isinstance(var.data, ManifestArray):
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
            append_dim=append_dim,
        )
    else:
        raise ValueError(
            "Cannot write non-virtual variables as virtual variables to Icechunk stores"
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
):  # -> "Array":
    existing_array = group[name]
    new_shape = list(existing_array.shape)
    new_shape[append_axis] += var.shape[append_axis]
    return existing_array.resize(tuple(new_shape))


def get_axis(
    dims: list[str],
    dim_name: str,
) -> int:
    return dims.index(dim_name)


def _check_compatible_arrays(
    ma: ManifestArray, existing_array: zarr.core.array.Array, append_axis: int
):
    manifest_api._check_same_dtypes([ma.dtype, existing_array.dtype])
    # this is kind of gross - _v3_codec_pipeline returns a tuple
    # Question: Does anything need to be done to apply the codecs to the new manifest array?
    manifest_api._check_same_codecs(
        [list(ma.zarray._v3_codec_pipeline()), existing_array.metadata.codecs]
    )
    manifest_api._check_same_chunk_shapes([ma.chunks, existing_array.chunks])
    manifest_api._check_same_ndims([ma.ndim, existing_array.ndim])
    arr_shapes = [ma.shape, existing_array.shape]
    manifest_api._check_same_shapes_except_on_concat_axis(arr_shapes, append_axis)


def check_compatible_encodings(encoding1, encoding2):
    for key, value in encoding1.items():
        if key in encoding2:
            if encoding2[key] != value:
                raise ValueError(
                    f"Cannot concatenate arrays with different values for encoding key {key}: {encoding2[key]} != {value}"
                )


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

    dims = var.dims
    append_axis, existing_num_chunks, arr = None, None, None
    if mode == "a" and append_dim in dims:
        existing_array = group[name]
        append_axis = get_axis(dims, append_dim)
        # check if arrays can be concatenated
        check_compatible_encodings(var.encoding, existing_array.attrs)
        _check_compatible_arrays(ma, existing_array, append_axis)

        # determine number of existing chunks along the append axis
        existing_num_chunks = num_chunks(
            array=group[name],
            axis=append_axis,
        )

        # resize the array
        arr = resize_array(
            group=group,
            name=name,
            var=var,
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
            # TODO fill_value?
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
    index: np.nditer.multi_index,
    append_axis: Optional[int] = None,
    existing_num_chunks: Optional[int] = None,
) -> str:
    if append_axis is not None:
        list_index = list(index)
        # Offset by the number of existing chunks on the append axis
        list_index[append_axis] += existing_num_chunks
        index = tuple(list_index)
    return "/".join(str(i) for i in index)


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
    # Aimee: the manifest (and it's corresponding paths, offsets and lengths, already has the shape of the datacube's chunks
    # so we want to increment the resulting multi index
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
