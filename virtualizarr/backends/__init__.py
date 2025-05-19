from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore

from virtualizarr.backends.dmrpp import backend as DMRPPBackend
from virtualizarr.backends.fits import backend as FITSBackend
from virtualizarr.backends.kerchunk import backend as KerchunkBackend
from virtualizarr.backends.zarr import backend as ZarrBackend

__all__ = [
    "DMRPPBackend",
    "FITSBackend",
    "KerchunkBackend",
    "ZarrBackend",
]

@runtime_checkable
class Backend(Protocol):
    def __call__(
        filepath: str,
        object_reader: ObjectStore,
    ) -> ManifestStore: ...
