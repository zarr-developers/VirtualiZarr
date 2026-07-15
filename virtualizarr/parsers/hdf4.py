from collections.abc import Iterable
from pathlib import Path

from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.types.kerchunk import KerchunkStoreRefs


class HDF4Parser:
    """Create a [ManifestStore][virtualizarr.manifests.ManifestStore] from an HDF4 file.

    Parameters
    ----------
    group
        The group within the file to be used as the Zarr root group for the ManifestStore.
    skip_variables
        Variables in the file that will be ignored when creating the ManifestStore.
    reader_options
        Configuration options used internally for kerchunk's fsspec backend.
    """

    def __init__(
        self,
        group: str | None = None,
        skip_variables: Iterable[str] | None = None,
        reader_options: dict | None = None,
    ):
        self.group = group
        self.skip_variables = skip_variables
        self.reader_options = reader_options or {}

    def __call__(
        self,
        url: str,
        registry: ObjectStoreRegistry,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given HDF4 file to produce a VirtualiZarr ManifestStore.

        Parameters
        ----------
        url
            The URL of the input HDF4 file (e.g., "s3://bucket/file.hdf").
        registry
            An [ObjectStoreRegistry][obspec_utils.registry.ObjectStoreRegistry] for resolving urls and reading data.

        Returns
        -------
        ManifestStore
            A ManifestStore that provides a Zarr representation of the parsed HDF4 file.
        """

        from kerchunk.hdf4 import HDF4ToZarr

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs(
            {"refs": HDF4ToZarr(url, **self.reader_options).translate()}
        )

        manifestgroup = manifestgroup_from_kerchunk_refs(
            refs,
            group=self.group,
            skip_variables=self.skip_variables,
            fs_root=Path.cwd().as_uri(),
        )
        return ManifestStore(group=manifestgroup, registry=registry)
