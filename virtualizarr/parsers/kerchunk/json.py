from collections.abc import Iterable

import ujson

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.registry import ObjectStoreRegistry


class KerchunkJSONParser:
    def __init__(
        self,
        group: str | None = None,
        fs_root: str | None = None,
        skip_variables: Iterable[str] | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the
        `__call__` method.

        Parameters
        ----------
        group
            The group within the Kerchunk JSON to be used as the Zarr root group for the ManifestStore.
        fs_root
            The qualifier to be used for Kerchunk chunk references containing relative paths.
        skip_variables
            Variables in the Kerchunk JSON that will be ignored when creating the ManifestStore.
        """

        self.group = group
        self.fs_root = fs_root
        self.skip_variables = skip_variables

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given Kerchunk JSON to produce a
        VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input Kerchunk JSON (e.g., "s3://bucket/kerchunk.json").
        registry
            An [ObjectStoreRegistry][virtualizarr.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the parsed Kerchunk JSON.
        """
        store, path_after_prefix = registry.resolve(url)

        # we need the whole thing so just get the entire contents in one request
        resp = store.get(path_after_prefix)
        content = resp.bytes().to_bytes()
        refs = ujson.loads(content)

        manifestgroup = manifestgroup_from_kerchunk_refs(
            refs,
            group=self.group,
            fs_root=self.fs_root,
            skip_variables=self.skip_variables,
        )
        return ManifestStore(group=manifestgroup, registry=registry)
