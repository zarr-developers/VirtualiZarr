from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.dmrpp import Parser as DMRPPParser
from virtualizarr.parsers.fits import Parser as FITSParser
from virtualizarr.parsers.hdf.hdf import Parser as HDFParser
from virtualizarr.parsers.kerchunk import Parser as KerchunkParser
from virtualizarr.parsers.kerchunk_parquet import Parser as KerchunkParquetParser
from virtualizarr.parsers.netcdf3 import Parser as NetCDF3Parser
from virtualizarr.parsers.zarr import Parser as ZarrParser

__all__ = [
    "DMRPPParser",
    "FITSParser",
    "HDFParser",
    "NetCDF3Parser",
    "KerchunkParser",
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
