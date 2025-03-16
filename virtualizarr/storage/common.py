from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from xarray import DataArray, Dataset
from xarray.backends.api import DATAARRAY_VARIABLE
from zarr.core.buffer import Buffer, default_buffer_prototype

from virtualizarr.types.general import T_Xarray
from virtualizarr.vendor.zarr.metadata import dict_to_buffer


@dataclass
class StoreRequest:
    """Dataclass for matching a key to the store instance"""

    store_id: str
    """The ID of a store."""
    key: str
    """The key within the store to request."""


@dataclass
class ManifestIndex:
    """Dataclass for indexing into ChunkManifests"""

    variable: str
    """Variable to extract keys, offsets, and lengths from."""
    indexes: tuple[int, ...]
    """Index of specific chunk within the ChunkManifest."""


async def list_dir_from_xr_obj(vd: T_Xarray, prefix: str) -> AsyncGenerator[str]:
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
    if prefix:
        raise NotImplementedError
    yield "zarr.json"
    if isinstance(vd, Dataset):
        for v in vd.variables:
            yield v
    if isinstance(vd, DataArray):
        yield DATAARRAY_VARIABLE
        for c in vd.coords:
            yield c


def get_zarr_metadata(vd: T_Xarray, key: str) -> Buffer:
    # If requesting the root metadata, return the standard group metadata with additional dataset specific attributes
    if key == "zarr.json":
        metadata = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": vd.attrs,
        }
        return dict_to_buffer(metadata, prototype=default_buffer_prototype())
    # Handle metadata for data variable within a DataArray
    elif key == "__xarray_dataarray_variable__/zarr.json":
        metadata = vd.data.metadata.to_dict()
        metadata["attributes"] = vd.attrs
    # Handle metadata for variables within Datasets
    else:
        var, _ = key.split("/")
        metadata = vd[var].data.metadata.to_dict()
        metadata["attributes"] = vd[var].attrs
        if not metadata.get("dimension_names", None):
            metadata["dimension_names"] = vd[var].dims
    return dict_to_buffer(metadata, prototype=default_buffer_prototype())


def parse_manifest_index(key: str) -> ManifestIndex:
    parts = key.split("/")
    var = parts[0]
    # Assume "c" is the second part
    # TODO: Handle scalar array case with "c" holds the data
    indexes = tuple(int(ind) for ind in parts[2:])
    return ManifestIndex(variable=var, indexes=indexes)


def find_matching_store(stores: dict[str, Any], request_key: str) -> StoreRequest:
    """
    Find which key in a dictionary matches the beginning of a given URI string.

    Parameters:
    -----------
    stores : dict
        A dictionary with URI prefixes for different stores as keys
    request_key : str
        A string to match against the stores dictionary keys

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
