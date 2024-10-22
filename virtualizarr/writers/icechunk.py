from typing import TYPE_CHECKING, cast

import numpy as np
from xarray import Dataset
from xarray.backends.zarr import encode_zarr_attr_value
from xarray.core.variable import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
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


def dataset_to_icechunk(ds: Dataset, store: "IcechunkStore") -> None:
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
    )


def write_variables_to_icechunk_group(
    variables,
    attrs,
    store,
    group,
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
    ds.to_zarr(store, zarr_format=3, consolidated=False, mode="a")

    # Then finish by writing the virtual variables to the same group
    for name, var in virtual_variables.items():
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
        )


def write_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: Variable,
) -> None:
    """Write a single (possibly virtual) variable into an icechunk store"""
    if isinstance(var.data, ManifestArray):
        write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
        )
    else:
        raise ValueError(
            "Cannot write non-virtual variables as virtual variables to Icechunk stores"
        )


def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: "Group",
    name: str,
    var: Variable,
) -> None:
    """Write a single virtual variable into an icechunk store"""
    ma = cast(ManifestArray, var.data)
    zarray = ma.zarray

    # creates array if it doesn't already exist
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
    arr.attrs["_ARRAY_DIMENSIONS"] = encode_zarr_attr_value(var.dims)

    _encoding_keys = {"_FillValue", "missing_value", "scale_factor", "add_offset"}
    for k, v in var.encoding.items():
        if k in _encoding_keys:
            arr.attrs[k] = encode_zarr_attr_value(v)

    write_manifest_virtual_refs(
        store=store,
        group=group,
        arr_name=name,
        manifest=ma.manifest,
    )


def write_manifest_virtual_refs(
    store: "IcechunkStore",
    group: "Group",
    arr_name: str,
    manifest: ChunkManifest,
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
        index = it.multi_index
        chunk_key = "/".join(str(i) for i in index)

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
