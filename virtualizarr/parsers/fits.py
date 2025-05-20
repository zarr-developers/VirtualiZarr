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
        drop_variables: Iterable[str] | None = None,
        reader_options: Optional[dict] = None,
    ):
        self.group = group
        self.drop_variables = drop_variables
        self.reader_options = reader_options

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:

        from kerchunk.fits import process_file
        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs({"refs": process_file(file_url, **self.reader_options)})

        manifeststore = manifeststore_from_kerchunk_refs(
            refs,
            group=self.group,
            drop_variables=self.drop_variables,
            fs_root=Path.cwd().as_uri(),
        )

        return manifeststore
