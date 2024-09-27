from typing import TYPE_CHECKING

import numpy as np
from xarray import Dataset

from virtualizarr.manifests import ManifestArray

if TYPE_CHECKING:
    import IcechunkStore


def dataset_to_icechunk(ds: Dataset, store: "IcechunkStore") -> None:
    """
    Write an xarray dataset whose variables wrap ManifestArrays to an Icechunk store.

    Currently requires all variables to be backed by ManifestArray objects.

    Parameters
    ----------
    ds: xr.Dataset
    store: IcechunkStore
    """

    # TODO write group metadata

    for name, var in ds.variables.items():
        if isinstance(var.data, ManifestArray):
            write_manifestarray_to_icechunk(
                store=store,
                # TODO is this right?
                group='root',
                arr_name=name,
                ma=var.data,
            )
        else:
            # TODO write loadable data as normal zarr chunks
            raise NotImplementedError()

    return None


def write_manifestarray_to_icechunk(
    store: "IcechunkStore", 
    group: str, 
    arr_name: str, 
    ma: ManifestArray,
) -> None:

    manifest = ma.manifest

    # TODO how do we set the other zarr attributes? i.e. the .zarray information?

    # loop over every reference in the ChunkManifest for that array
    # TODO this should be replaced with something more efficient that sets all (new) references for the array at once
    # but Icechunk need to expose a suitable API first
    for entry in np.nditer(
        [manifest._paths, manifest._offsets, manifest._lengths], 
        flags=['multi_index'],
    ):
        # set each reference individually
        store.set_virtual_ref(
            # TODO make sure this key is the correct format
            key=f"{group}/{arr_name}/{entry.index}", # your (0,1,2) tuple
            location=entry[0],  # filepath for this element
            offset=entry[1],  # offset for this element
            length=entry[2],  # length for this element
        )
