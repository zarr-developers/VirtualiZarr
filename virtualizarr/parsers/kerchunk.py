from typing import Iterable

import ujson
from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.store import ObjectStoreRegistry, get_store_prefix
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs
from virtualizarr.utils import ObstoreReader


class Parser:
    def __init__(
        self,
        group: str | None = None,
        fs_root: str | None = None,
        drop_variables: Iterable[str] | None = None,
        store_registry: ObjectStoreRegistry | None = None,
    ):
        self.group = group
        self.fs_root = fs_root
        self.drop_variables = drop_variables
        self.store_registry = store_registry

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        reader = ObstoreReader(store=object_store, path=file_url)

        # JSON has no magic bytes, but the Kerchunk version 1 spec starts with 'version':
        # https://fsspec.github.io/kerchunk/spec.html
        error_message = "The input Kerchunk reference did not seem to be in Kerchunk's JSON or Parquet spec: https://fsspec.github.io/kerchunk/spec.html. If your Kerchunk generated references are saved in parquet format, make sure the file extension is `.parquet`. The Kerchunk format autodetection is quite flaky, so if your reference matches the Kerchunk spec feel free to open an issue: https://github.com/zarr-developers/VirtualiZarr/issues"
        try:
            has_version = reader.read(9).startswith(b'{"version')
        except OSError:
            raise ValueError(error_message)
        if has_version:
            reader.seek(0)
            content = reader.readall().decode()
            refs = ujson.loads(content)
            if self.store_registry is None: 
                unique_paths = {
                    v[0] for v in refs['refs'].values()
                    if isinstance(v, list) and isinstance(v[0], str)
                }
                stores = {}
                for path in unique_paths:
                    stores[get_store_prefix(path)] = object_store
                registry = ObjectStoreRegistry(stores=stores)
            else:
                registry = self.store_registry
            manifeststore = manifeststore_from_kerchunk_refs(
                refs,
                group=self.group,
                fs_root=self.fs_root,
                drop_variables=self.drop_variables,
                store_registry=registry,
            )
            return manifeststore
        else:
            raise ValueError(error_message)
