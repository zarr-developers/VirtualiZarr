from __future__ import annotations

from typing import Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore

__all__ = ["Parser"]


@runtime_checkable
class Parser(Protocol):
    def __call__(
        self,
        filepath: str,
        object_reader: ObjectStore,
    ) -> ManifestStore: ...
