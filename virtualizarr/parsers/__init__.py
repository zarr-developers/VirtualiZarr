from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.dmrpp import DMRPPParser
from virtualizarr.parsers.fits import FITSParser
from virtualizarr.parsers.hdf.hdf import HDFParser
from virtualizarr.parsers.kerchunk_json import KerchunkJSONParser
from virtualizarr.parsers.kerchunk_parquet import KerchunkParquetParser
from virtualizarr.parsers.netcdf3 import NetCDF3Parser
from virtualizarr.parsers.zarr import ZarrParser

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
    Parse the contents of a given file to produce a ManifestStore.

    Effectively maps the contents of the file (e.g. metadata, compression codecs, chunk byte offsets) to the Zarr data model.

    Parameters
    ----------
    file_url
        The URI or path to the input file (e.g., "s3://bucket/file.nc").
    object_store
        An obstore ObjectStore instance for accessing the file specified in the `file_url` parameter.

    Returns
    -------
    ManifestStore
        A ManifestStore which provides a Zarr representation of the parsed file.
    """
