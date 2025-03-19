from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, TypeAlias

from obstore.store import ObjectStore
from zarr.core.buffer import Buffer, default_buffer_prototype

from virtualizarr.manifests.group import ManifestDict, ManifestGroup
from virtualizarr.vendor.zarr.metadata import dict_to_buffer

StoreDict: TypeAlias = dict[str, ObjectStore]


@dataclass
class StoreRequest:
    """Dataclass for matching a key to the store instance"""

    store_id: str
    """The ID of a store."""
    key: str
    """The key within the store to request."""


async def list_dir_from_manifest_dict(
    manifest_arrays: ManifestDict, prefix: str
) -> AsyncGenerator[str]:
    """Create the expected results for Zarr's `store.list_dir()` from an Xarray DataArrray or Dataset

    Parameters
    ----------
    vd : xarray DataArray or Dataset
    prefix : str


    Returns
    -------
    AsyncIterator[str]
    """
    # Start with expected group level metadata
    raise NotImplementedError


def get_zarr_metadata(manifest_group: ManifestGroup, key: str) -> Buffer:
    """
    Generate the expected Zarr V3 metadata from a virtual dataset.

    Group metadata is returned for all Datasets and Array metadata
    is returned for all DataArrays.

    Combines the ManifestArray metadata with the attrs from the DataArray
    and adds `dimension_names` for all arrays if not already provided.

    Parameters
    ----------
    vd : xarray DataArray or Dataset
    key : str

    Returns
    -------
    Buffer
    """
    # If requesting the root metadata, return the standard group metadata with additional dataset specific attributes

    if key == "zarr.json":
        metadata = manifest_group._metadata.to_dict()
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())
    else:
        var, _ = key.split("/")
        metadata = manifest_group._manifest_dict[var].metadata.to_dict()
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())


def parse_manifest_index(key: str) -> tuple[str, tuple[int, ...]]:
    """
    Splits `key` provided to a zarr store into the variable indicated
    by the first part and the chunk index from the 3rd through last parts,
    which can be used to index into the ndarrays containing paths, offsets,
    and lengths in ManifestArrays.

    Currently only works for 1d+ arrays with a tree depth of one from the
    root Zarr group.

    Parameters
    ----------
    key : str

    Returns
    -------
    ManifestIndex
    """
    parts = key.split("/")
    var = parts[0]
    # Assume "c" is the second part
    # TODO: Handle scalar array case with "c" holds the data
    indexes = tuple(int(ind) for ind in parts[2:])
    return var, indexes


def find_matching_store(stores: dict[str, Any], request_key: str) -> StoreRequest:
    """
    Find the matching store based on the store keys and the beginning of the URI strings,
    to fetch data from the appropriately configured ObjectStore.

    Parameters:
    -----------
    stores : dict
        A dictionary with URI prefixes for different stores as keys
    request_key : str
        A string to match against the dictionary keys

    Returns:
    --------
    StoreRequest
    """
    # Sort keys by length in descending order to ensure longer, more specific matches take precedence
    sorted_keys = sorted(stores.keys(), key=len, reverse=True)

    # Check each key to see if it's a prefix of the uri_string
    for key in sorted_keys:
        if request_key.startswith(key):
            return StoreRequest(store_id=key, key=request_key[len(key) :])
    # if no match is found, raise an error
    raise ValueError(
        f"Expected the one of stores.keys() to match the data prefix, got {stores.keys()} and {request_key}"
    )
