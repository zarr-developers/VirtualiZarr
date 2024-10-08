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


def _fsspec_openfile_from_filepath(
    *,
    filepath: str,
    reader_options: Optional[dict] = None,
) -> OpenFileType:
    """Converts input filepath to fsspec openfile object.

    Parameters
    ----------
    filepath : str
        Input filepath
    reader_options : dict, optional
        Dict containing kwargs to pass to file opener, by default {}

    Returns
    -------
    OpenFileType
        An open file-like object, specific to the protocol supplied in filepath.

    Raises
    ------
    NotImplementedError
        Raises a Not Implemented Error if filepath protocol is not supported.
    """

    import fsspec
    from upath import UPath

    universal_filepath = UPath(filepath)
    protocol = universal_filepath.protocol

    if reader_options is None:
        reader_options = {}

    storage_options = reader_options.get("storage_options", {})  # type: ignore

    fpath = fsspec.filesystem(protocol, **storage_options).open(filepath)

    return fpath
