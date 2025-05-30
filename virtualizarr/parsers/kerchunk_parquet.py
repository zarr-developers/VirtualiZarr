from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional, Union

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.store import ObjectStoreRegistry, get_store_prefix
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs
from virtualizarr.types.kerchunk import (
    KerchunkStoreRefs,
)

if TYPE_CHECKING:
    import fsspec
    import fsspec.core
    import fsspec.spec
    import upath
    from obstore.store import ObjectStore

    # See pangeo_forge_recipes.storage
    OpenFileType = Union[
        fsspec.core.OpenFile, fsspec.spec.AbstractBufferedFile, io.IOBase
    ]


class Parser:
    def __init__(
        self,
        group: str | None = None,
        fs_root: str | None = None,
        skip_variables: Iterable[str] | None = None,
        reader_options: Optional[dict] = None,
    ):
        """
        Instantiate a parser with parser-specific parameters that can be used in the __call__ method.
        Parameters:
            group (str): The group within the file to be used as the Zarr root group for the ManifestStore.
            fs_root (str): The qualifier to be used for kerchunk references containing relative paths.
            skip_variables (Iterable[str]): Variables in the file that will be ignored when creating the ManifestStore.
            reader_options: (dict): Configuration options used internally for the fsspec backend.
        """

        self.group = group
        self.fs_root = fs_root
        self.skip_variables = skip_variables
        self.reader_options = reader_options

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        """
        Parse the metadata and byte offsets from a given file to product a
        VirtualiZarr ManifestStore.

        Parameters:
            file_url (str): The URI or path to the input parquet directory (e.g., "s3://bucket/file.parq").
            object_store (ObjectStore): An obstore ObjectStore instance for accessing the file specified in the file_url parameter.

        Returns:
            ManifestStore: A ManifestStore which provides a Zarr representation of the parsed file.
        """

        # The kerchunk .parquet storage format isn't actually a parquet, but a directory that contains named parquets for each group/variable.
        fs = _FsspecFSFromFilepath(
            filepath=file_url, reader_options=self.reader_options
        )
        from fsspec.implementations.reference import LazyReferenceMapper

        lrm = LazyReferenceMapper(file_url, fs.fs)

        # build reference dict from KV pairs in LazyReferenceMapper
        # is there a better / more performant way to extract this?
        array_refs = {k: lrm[k] for k in lrm.keys()}
        full_reference = {"refs": array_refs}
        refs = KerchunkStoreRefs(full_reference)
        registry = ObjectStoreRegistry({get_store_prefix(file_url): object_store})
        manifeststore = manifeststore_from_kerchunk_refs(
            refs,
            group=self.group,
            fs_root=self.fs_root,
            skip_variables=self.skip_variables,
            store_registry=registry,
        )
        return manifeststore


@dataclass
class _FsspecFSFromFilepath:
    """Class to create fsspec Filesystem from input filepath.

    Parameters
    ----------
    filepath : str
        Input filepath
    reader_options : dict, optional
        dict containing kwargs to pass to file opener, by default {}
    fs : Option | None
        The fsspec filesystem object, created in __post_init__

    """

    filepath: str
    reader_options: Optional[dict] = field(default_factory=dict)
    fs: fsspec.AbstractFileSystem = field(init=False)
    upath: upath.core.UPath = field(init=False)

    def open_file(self) -> OpenFileType:
        """Calls `.open` on fsspec.Filesystem instantiation using self.filepath as an input.

        Returns
        -------
        OpenFileType
            file opened with fsspec
        """
        return self.fs.open(self.filepath)

    def read_bytes(self, bytes: int) -> bytes:
        with self.open_file() as of:
            return of.read(bytes)

    def get_mapper(self):
        """Returns a mapper for use with Zarr"""
        return self.fs.get_mapper(self.filepath)

    def __post_init__(self) -> None:
        """Initialize the fsspec filesystem object"""
        import fsspec
        from upath import UPath

        if not isinstance(self.filepath, UPath):
            upath = UPath(self.filepath)

        self.upath = upath
        self.protocol = upath.protocol

        self.reader_options = self.reader_options or {}
        storage_options = self.reader_options.get("storage_options", {})  # type: ignore

        self.fs = fsspec.filesystem(self.protocol, **storage_options)
