from __future__ import annotations

import io
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import fsspec.core
    import fsspec.spec

    # See pangeo_forge_recipes.storage
    OpenFileType = Union[
        fsspec.core.OpenFile, fsspec.spec.AbstractBufferedFile, io.IOBase
    ]


from dataclasses import dataclass, field


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

    def open_file(self) -> OpenFileType:
        """Calls `.open` on fsspec.Filesystem instantiation using self.filepath as an input.

        Returns
        -------
        OpenFileType
            file opened with fsspec
        """
        return self.fs.open(self.filepath)

    def __post_init__(self) -> None:
        """Initialize the fsspec filesystem object"""
        import fsspec
        from upath import UPath

        universal_filepath = UPath(self.filepath)
        protocol = universal_filepath.protocol

        self.reader_options = self.reader_options or {}
        storage_options = self.reader_options.get("storage_options", {})  # type: ignore

        self.fs = fsspec.filesystem(protocol, **storage_options)
