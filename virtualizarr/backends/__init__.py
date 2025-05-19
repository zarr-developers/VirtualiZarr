from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore

from .dmrpp import backend as DMRPPBackend
from .kerchunk import backend as KerchunkBackend
from .zarr import backend as ZarrBackend

__all__ = [
    "DMRPPBackend",
    "KerchunkBackend",
    "ZarrBackend",
]

@runtime_checkable
class Backend(Protocol):
    def __call__(
        filepath: str,
        object_reader: ObjectStore,
    ) -> ManifestStore: ...
