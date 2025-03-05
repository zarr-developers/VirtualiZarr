"""
Virtual dataset operations.

This module contains functions for working with virtual datasets.
"""

import xarray as xr
from config import drop_vars
from repo import open_or_create_repo

from virtualizarr import open_virtual_dataset


def map_open_virtual_dataset(uri, open_args: dict = {}):
    """
    Map function to open virtual datasets.

    Args:
        uri: The URI of the virtual dataset

    Returns:
        A virtual dataset
    """
    vds = open_virtual_dataset(
        uri,
        indexes={},
        **open_args,
    )
    return vds.drop_vars(drop_vars, errors="ignore")


def concat_virtual_datasets(results):
    """
    Reduce to concat virtual datasets.

    Args:
        results: A list of virtual datasets

    Returns:
        A concatenated virtual dataset
    """
    combined_vds = xr.concat(
        results,
        dim="time",
        coords="minimal",
        compat="override",
        combine_attrs="override",
    )
    return combined_vds


def write_virtual_results_to_icechunk(
    virtual_ds, start_date: str, end_date: str, append_dim: str = None
):
    """
    Write virtual dataset results to IceChunk.

    Args:
        virtual_ds: The virtual dataset to write
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        append_dim: The dimension to append to (optional)

    Returns:
        The commit ID
    """
    repo = open_or_create_repo()
    session = repo.writable_session("main")

    # Check if store is already populated
    with session.allow_pickling():
        if append_dim:
            # Only use append_dim if store already has data
            virtual_ds.virtualize.to_icechunk(session.store, append_dim=append_dim)
        else:
            # If we can't check or there's an error, assume store is empty
            virtual_ds.virtualize.to_icechunk(session.store)

    return session.commit(f"Commit data {start_date} to {end_date}")


def concat_and_write_virtual_datasets(
    results, start_date: str, end_date: str, append_dim: str = None
):
    """
    Reduce to concat virtual datasets and write to icechunk.

    Args:
        results: A list of virtual datasets
        start_date: The start date in YYYY-MM-DD format
        end_date: The end date in YYYY-MM-DD format
        append_dim: The dimension to append to (optional)

    Returns:
        The commit ID
    """
    combined_vds = concat_virtual_datasets(results)
    return write_virtual_results_to_icechunk(
        combined_vds, start_date, end_date, append_dim
    )
