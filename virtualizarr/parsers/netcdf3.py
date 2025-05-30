from pathlib import Path
from typing import Iterable, Optional

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs


class Parser:
    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        reader_options: Optional[dict] = {},
    ):
        self.group = group
        self.skip_variables = skip_variables
        self.reader_options = reader_options

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        from kerchunk.netCDF3 import NetCDF3ToZarr

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = NetCDF3ToZarr(
            file_url, inline_threshold=0, **self.reader_options
        ).translate()

        manifeststore = manifeststore_from_kerchunk_refs(
            refs,
            group=self.group,
            skip_variables=self.skip_variables,
            fs_root=Path.cwd().as_uri(),
        )

        return manifeststore
