from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore

if TYPE_CHECKING:
    from obstore import ReadableFile

@runtime_checkable
class Backend(Protocol):
    def __call__(
        self,
        filepath: str,
        file: ReadableFile,
        object_reader: ObjectStore,
        **kwargs: Any
    ) -> ManifestStore: ...


