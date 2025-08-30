from __future__ import annotations

import copy
import importlib
import io
import json
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence, Union

import obstore as obs
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata
from zarr.dtype import data_type_registry

from virtualizarr.codecs import get_codec_config, zarr_codec_config_to_v2
from virtualizarr.types.kerchunk import KerchunkStoreRefs

# taken from zarr.core.common
JSON = str | int | float | Mapping[str, "JSON"] | Sequence["JSON"] | None


if TYPE_CHECKING:
    import fsspec.core
    import fsspec.spec
    from obstore import ReadableFile
    from obstore.store import ObjectStore

    # See pangeo_forge_recipes.storage
    OpenFileType = Union[
        fsspec.core.OpenFile, fsspec.spec.AbstractBufferedFile, io.IOBase
    ]


class ObstoreReader:
    _reader: ReadableFile

    def __init__(self, store: ObjectStore, path: str) -> None:
        """
        Create an obstore file reader that implements the read, readall, seek, and tell methods, which
        can be used in libraries that expect file-like objects.

        Parameters
        ----------
        store
            [ObjectStore][obstore.store.ObjectStore] for reading the file.
        path
            The path to the file within the store. This should not include the prefix.
        """
        self._reader = obs.open_reader(store, path)

    def read(self, size: int, /) -> bytes:
        return self._reader.read(size).to_bytes()

    def readall(self) -> bytes:
        return self._reader.read().to_bytes()

    def seek(self, offset: int, whence: int = 0, /):
        # TODO: Check on default for whence
        return self._reader.seek(offset, whence)

    def tell(self) -> int:
        return self._reader.tell()


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
    v3_metadata
        The metadata object in v3 format.
    fill_value
        Override the fill value from v3 metadata.

    Returns
    -------
    ArrayV2Metadata
        The metadata object in v2 format.
    """

    # TODO: Check that all ArrayBytesCodecs should in fact be excluded for V2 metadata storage.
    v2_codecs = [
        zarr_codec_config_to_v2(get_codec_config(codec))
        for codec in v3_metadata.codecs
        if not isinstance(codec, ArrayBytesCodec)
    ]
    # TODO: Remove convert_v3_to_v2_metadata and always encode V3 metadata.
    # This logic is based on the (default) Bytes codec's endian property,
    # but other codec pipelines could store endianness elsewhere.
    big_endian = any(
        isinstance(codec, ArrayBytesCodec)
        and getattr(codec, "endian", None) is not None
        and codec.endian.value == "big"  # type: ignore[attr-defined]
        for codec in v3_metadata.codecs
    )
    if big_endian:
        na_dtype = v3_metadata.data_type.to_native_dtype().newbyteorder(">")
        dtype = data_type_registry.match_dtype(dtype=na_dtype)
    else:
        dtype = v3_metadata.data_type
    v2_metadata = ArrayV2Metadata(
        shape=v3_metadata.shape,
        dtype=dtype,
        chunks=v3_metadata.chunks,
        fill_value=fill_value or v3_metadata.fill_value,
        filters=v2_codecs
        if v2_codecs
        else None,  # Do not pass an empty list to ArrayV2Metadata
        compressor=None,
        order="C",
        attributes=v3_metadata.attributes,
        dimension_separator=".",  # Assuming '.' as default dimension separator
    )
    return v2_metadata


def kerchunk_refs_as_json(refs: KerchunkStoreRefs) -> JSON:
    """
    Normalizes all Kerchunk references into true JSON all the way down.

    See https://github.com/zarr-developers/VirtualiZarr/issues/679 for context as to why this is needed.
    """

    normalized_result: dict[str, JSON] = copy.deepcopy(refs)
    v0_refs: dict[str, JSON] = refs["refs"]

    for k, v in v0_refs.items():
        # check for strings because the value could be for a chunk, in which case it is already a list like ["/test.nc", 6144, 48]
        # this is a rather fragile way to discover if we're looking at a chunk key or not, but it should work...
        if isinstance(v, str):
            normalized_result["refs"][k] = json.loads(v)  # type: ignore[index]

    return normalized_result
