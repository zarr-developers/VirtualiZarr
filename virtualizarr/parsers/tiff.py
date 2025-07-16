from collections.abc import Iterable
from pathlib import Path

from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.registry import ObjectStoreRegistry
from virtualizarr.types.kerchunk import KerchunkStoreRefs


class Parser:
    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        remote_options: dict | None = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the
        `__call__` method.

        Parameters
        ----------
        group
            The group within the file to be used as the Zarr root group for the ManifestStore.
        skip_variables
            Variables in the file that will be ignored when creating the ManifestStore.
        remote_options
            Configuration options used internally for kerchunk's fsspec backend
        """

        self.group = group
        self.skip_variables = skip_variables
        self.remote_options = remote_options or {}

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given file to product a VirtualiZarr ManifestStore.

        Parameters
        ----------
        file_url
            The URI or path to the input file (e.g., "s3://bucket/file.tiff").
        object_store
            An obstore ObjectStore instance for accessing the file specified in the
            `file_url` parameter.

        Returns
        -------
        ManifestStore
            A ManifestStore which provides a Zarr representation of the parsed file.
        """

        from kerchunk.tiff import tiff_to_zarr

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs(
            {"refs": tiff_to_zarr(file_url, **self.remote_options)}
        )

        manifestgroup = manifestgroup_from_kerchunk_refs(
            refs,
            group=self.group,
            skip_variables=self.skip_variables,
            fs_root=Path.cwd().as_uri(),
        )
        registry = ObjectStoreRegistry()
        registry.register(file_url, object_store)

        return ManifestStore(group=manifestgroup, store_registry=registry)
