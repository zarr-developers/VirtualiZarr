from __future__ import annotations

import importlib
import io
from typing import TYPE_CHECKING, Iterable, Optional, Union

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

    def read_bytes(self, bytes: int) -> bytes:
        with self.open_file() as of:
            return of.read(bytes)

    def __post_init__(self) -> None:
        """Initialize the fsspec filesystem object"""
        import fsspec
        from upath import UPath

        universal_filepath = UPath(self.filepath)
        protocol = universal_filepath.protocol

        self.reader_options = self.reader_options or {}
        storage_options = self.reader_options.get("storage_options", {})  # type: ignore

        self.fs = fsspec.filesystem(protocol, **storage_options)


def check_for_collisions(
    drop_variables: Iterable[str] | None,
    loadable_variables: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    if drop_variables is None:
        drop_variables = []
    elif isinstance(drop_variables, str):
        drop_variables = [drop_variables]
    else:
        drop_variables = list(drop_variables)

    if loadable_variables is None:
        loadable_variables = []
    elif isinstance(loadable_variables, str):
        loadable_variables = [loadable_variables]
    else:
        loadable_variables = list(loadable_variables)

    common = set(drop_variables).intersection(set(loadable_variables))
    if common:
        raise ValueError(f"Cannot both load and drop variables {common}")

    return drop_variables, loadable_variables


def soft_import(name: str, reason: str, strict: Optional[bool] = True):
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        if strict:
            raise ImportError(
                f"for {reason}, the {name} package is required. "
                f"Please install it via pip or conda."
            )
        else:
            return None
