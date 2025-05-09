from __future__ import annotations

import importlib
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from virtualizarr.codecs import extract_codecs, get_codec_config

if TYPE_CHECKING:
    import fsspec.core
    import fsspec.spec
    import upath
    from obstore import ReadableFile

    # See pangeo_forge_recipes.storage
    OpenFileType = Union[
        fsspec.core.OpenFile, fsspec.spec.AbstractBufferedFile, io.IOBase
    ]


class ObstoreReader:
    _reader: ReadableFile

    def __init__(self, file: ReadableFile) -> None:
        self._reader = file 

    def read(self, size: int, /) -> bytes:
        return self._reader.read(size).to_bytes()

    def seek(self, offset: int, whence: int = 0, /):
        # TODO: Check on default for whence
        return self._reader.seek(offset, whence)

    def tell(self) -> int:
        return self._reader.tell()


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


def ceildiv(a: int, b: int) -> int:
    """
    Ceiling division operator for integers.

    See https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(a // -b)


def determine_chunk_grid_shape(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[int, ...]:
    """Calculate the shape of the chunk grid based on array shape and chunk size."""
    return tuple(ceildiv(length, chunksize) for length, chunksize in zip(shape, chunks))


def convert_v3_to_v2_metadata(
    v3_metadata: ArrayV3Metadata, fill_value: Any = None
) -> ArrayV2Metadata:
    """
    Convert ArrayV3Metadata to ArrayV2Metadata.

    Parameters
    ----------
    v3_metadata : ArrayV3Metadata
        The metadata object in v3 format.
    fill_value : Any, optional
        Override the fill value from v3 metadata.

    Returns
    -------
    ArrayV2Metadata
        The metadata object in v2 format.
    """
    import warnings

    array_filters: tuple[ArrayArrayCodec, ...]
    bytes_compressors: tuple[BytesBytesCodec, ...]
    array_filters, _, bytes_compressors = extract_codecs(v3_metadata.codecs)
    # Handle compressor configuration
    compressor_config: dict[str, Any] | None = None
    if bytes_compressors:
        if len(bytes_compressors) > 1:
            warnings.warn(
                "Multiple compressors found in v3 metadata. Using the first compressor, "
                "others will be ignored. This may affect data compatibility.",
                UserWarning,
            )
        compressor_config = get_codec_config(bytes_compressors[0])

    # Handle filter configurations
    filter_configs = [get_codec_config(filter_) for filter_ in array_filters]

    v2_metadata = ArrayV2Metadata(
        shape=v3_metadata.shape,
        dtype=v3_metadata.data_type.to_numpy(),
        chunks=v3_metadata.chunks,
        fill_value=fill_value or v3_metadata.fill_value,
        compressor=compressor_config,
        filters=filter_configs,
        order="C",
        attributes=v3_metadata.attributes,
        dimension_separator=".",  # Assuming '.' as default dimension separator
    )
    return v2_metadata
