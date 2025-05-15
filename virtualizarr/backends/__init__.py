from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore

from .kerchunk import backend as KerchunkBackend

__all__ = [
    "KerchunkBackend"
]

@runtime_checkable
class Backend(Protocol):
    def __call__(
        filepath: str,
        object_reader: ObjectStore,
    ) -> ManifestStore: ...
