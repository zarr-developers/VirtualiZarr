from pathlib import Path
from typing import Iterable, Optional

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs
from virtualizarr.types.kerchunk import KerchunkStoreRefs


class Parser:
    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        remote_options: Optional[dict] = None,
    ):
        self.group = group
        self.skip_variables = skip_variables
        self.remote_options = remote_options

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        from kerchunk.tiff import tiff_to_zarr

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs(
            {"refs": tiff_to_zarr(file_url, **self.remote_options)}
        )

        manifeststore = manifeststore_from_kerchunk_refs(
            refs,
            group=self.group,
            skip_variables=self.skip_variables,
            fs_root=Path.cwd().as_uri(),
        )

        return manifeststore
