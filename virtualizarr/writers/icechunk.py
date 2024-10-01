import asyncio
from typing import TYPE_CHECKING

import numpy as np
from xarray import Dataset
from xarray.core.variable import Variable
from zarr import Group

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.zarr import encode_dtype

if TYPE_CHECKING:
    from icechunk import IcechunkStore


def dataset_to_icechunk(ds: Dataset, store: "IcechunkStore") -> None:
    """
    Write an xarray dataset whose variables wrap ManifestArrays to an Icechunk store.

    Currently requires all variables to be backed by ManifestArray objects.

    Parameters
    ----------
    ds: xr.Dataset
    store: IcechunkStore
    """
    from icechunk import IcechunkStore

    if not isinstance(store, IcechunkStore):
        raise TypeError(f"expected type IcechunkStore, but got type {type(store)}")

    # TODO should we check that the store supports writes at this point?

    # TODO only supports writing to the root group currently
    # TODO pass zarr_format kwarg?
    root_group = Group.from_store(store=store)

    # TODO this is Frozen, the API for setting attributes must be something else
    # root_group.attrs = ds.attrs
    for k, v in ds.attrs.items():
        root_group.attrs[k] = v

    asyncio.run(
        write_variables_to_icechunk_group(
            ds.variables,
            store=store,
            group=root_group,
        )
    )


async def write_variables_to_icechunk_group(
    variables,
    store,
    group,
):
    # we should be able to write references for each variable concurrently
    # TODO we could also write to multiple groups concurrently, i.e. in a future DataTree.to_zarr(icechunkstore)
    await asyncio.gather(
        *(
            write_variable_to_icechunk(
                store=store,
                group=group,
                name=name,
                var=var,
            )
            for name, var in variables.items()
        )
    )


async def write_variable_to_icechunk(
    store: "IcechunkStore",
    group: Group,
    name: str,
    var: Variable,
) -> None:
    """Write a single (possibly virtual) variable into an icechunk store"""

    if isinstance(var.data, ManifestArray):
        await write_virtual_variable_to_icechunk(
            store=store,
            group=group,
            name=name,
            var=var,
        )
    else:
        # TODO is writing loadable_variables just normal xarray ds.to_zarr?
        raise NotImplementedError()


async def write_virtual_variable_to_icechunk(
    store: "IcechunkStore",
    group: Group,
    name: str,
    var: Variable,
) -> None:
    """Write a single virtual variable into an icechunk store"""

    ma = var.data
    zarray = ma.zarray

    # TODO should I be checking that this array doesn't already exist? Or is that icechunks' job?
    arr = group.create_array(
        name=name,
        shape=zarray.shape,
        chunk_shape=zarray.chunks,
        dtype=encode_dtype(zarray.dtype),
        # TODO fill_value?
        # TODO order?
        # TODO zarr format?
        # TODO compressors?
    )

    # TODO it would be nice if we could assign directly to the .attrs property
    for k, v in var.attrs.items():
        arr.attrs[k] = v
    # TODO we should probably be doing some encoding of those attributes?
    arr.attrs["DIMENSION_NAMES"] = var.dims

    await write_manifest_virtual_refs(
        store=store,
        group=group,
        arr_name=name,
        manifest=ma.manifest,
    )


async def write_manifest_virtual_refs(
    store: "IcechunkStore",
    group: Group,
    arr_name: str,
    manifest: ChunkManifest,
) -> None:
    """Write all the virtual references for one array manifest at once."""

    key_prefix = f"{group.name}{arr_name}"

    # loop over every reference in the ChunkManifest for that array
    # TODO inefficient: this should be replaced with something that sets all (new) references for the array at once
    # but Icechunk need to expose a suitable API first
    it = np.nditer(
        [manifest._paths, manifest._offsets, manifest._lengths],
        flags=[
            "refs_ok",
            "multi_index",
            "c_index",  # TODO is "c_index" correct? what's the convention for zarr chunk keys?
        ],
        op_flags=[["readonly"]] * 3,
    )
    for path, offset, length in it:
        index = it.multi_index
        chunk_key = "/".join(str(i) for i in index)

        # TODO this needs to be awaited or something
        # set each reference individually
        await store.set_virtual_ref(
            # TODO it would be marginally neater if I could pass the group and name as separate args
            key=f"{key_prefix}/c/{chunk_key}",  # should be of form 'group/arr_name/c/0/1/2', where c stands for chunks
            location=path.item(),
            offset=offset.item(),
            length=length.item(),
        )
