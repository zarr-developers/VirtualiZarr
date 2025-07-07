from collections.abc import Iterable
from urllib.parse import urlparse

import obstore
import ujson
from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.utils import remove_prefix


class KerchunkJSONParser:
    def __init__(
        self,
        group: str | None = None,
        fs_root: str | None = None,
        skip_variables: Iterable[str] | None = None,
        store_registry: ObjectStoreRegistry | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the
        `__call__` method.

        Parameters
        ----------
        group
            The group within the file to be used as the Zarr root group for the ManifestStore.
        fs_root
            The qualifier to be used for kerchunk references containing relative paths.
        skip_variables
            Variables in the file that will be ignored when creating the ManifestStore.
        store_registry
            A user defined ObjectStoreRegistry to be used for reading data for kerchunk
            references contain paths to multiple locations.
        """

        self.group = group
        if fs_root:
            if not urlparse(fs_root).scheme:
                raise ValueError(
                    f"fs_root should include a valid scheme (e.g., `file://`), received `fs_root = '{fs_root}'`."
                )
        self.fs_root = fs_root
        self.skip_variables = skip_variables
        self.store_registry = store_registry

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given file to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        file_url
            The URI or path to the input file (e.g., "s3://bucket/kerchunk.json").
        object_store
            An obstore ObjectStore instance for accessing the file specified in the
            `file_url` parameter.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the parsed file.
        """

        path = remove_prefix(object_store, file_url)

        # we need the whole thing so just get the entire contents in one request
        resp = obstore.get(object_store, path)
        content = resp.bytes().to_bytes()
        refs = ujson.loads(content)

        # TODO: Deduplicate with KerchunkParquetParser
        if self.store_registry is None:
            unique_paths = {
                v[0]
                for v in refs["refs"].values()
                if isinstance(v, list) and isinstance(v[0], str)
            }
            registry = ObjectStoreRegistry()
            for path in unique_paths:
                parsed = urlparse(path)
                if self.fs_root and not parsed.scheme:
                    registry.register(f"{self.fs_root}{path}", object_store)
                elif not parsed.scheme:
                    raise ValueError(
                        f"Detected relative path `{path}`. Parser must have `fs_root` set for "
                        f"handling relative path. Parser has `fs_root = {self.fs_root}`."
                    )
                else:
                    registry.register(path, object_store)
        else:
            registry = self.store_registry

        manifestgroup = manifestgroup_from_kerchunk_refs(
            refs,
            group=self.group,
            fs_root=self.fs_root,
            skip_variables=self.skip_variables,
        )
        return ManifestStore(group=manifestgroup, store_registry=registry)
