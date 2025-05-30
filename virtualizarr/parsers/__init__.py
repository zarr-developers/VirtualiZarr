from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.dmrpp import Parser as DMRPPParser
from virtualizarr.parsers.fits import Parser as FITSParser
from virtualizarr.parsers.hdf.hdf import Parser as HDFParser
from virtualizarr.parsers.kerchunk_json import Parser as KerchunkJSONParser
from virtualizarr.parsers.kerchunk_parquet import Parser as KerchunkParquetParser
from virtualizarr.parsers.netcdf3 import Parser as NetCDF3Parser
from virtualizarr.parsers.zarr import Parser as ZarrParser

__all__ = [
    "DMRPPParser",
    "FITSParser",
    "HDFParser",
    "NetCDF3Parser",
    "KerchunkJSONParser",
    "KerchunkParquetParser",
    "ZarrParser",
]


@runtime_checkable
class Parser(Protocol):
    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore: ...

    """
    Parse the metadata and byte offsets from a given file to product a
    VirtualiZarr ManifestStore.

    Parameters:
        file_url (str): The URI or path to the input file (e.g., "s3://bucket/file.nc").
        object_store (ObjectStore): An obstore ObjectStore instance for accessing the file specified in the file_url parameter.

    Returns:
        ManifestStore: A ManifestStore which provides a Zarr representation of the parsed file.
    """
