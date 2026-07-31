import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr.manifests import ManifestStore
from virtualizarr.parsers.kerchunk.translator import manifestgroup_from_kerchunk_refs
from virtualizarr.types.kerchunk import KerchunkStoreRefs

# Serializes the monkeypatch in _positive_chunk_edges, which mutates zarr.Group
# for the duration of a single kerchunk translation.
_patch_lock = threading.Lock()


@contextmanager
def _positive_chunk_edges() -> Iterator[None]:
    """
    Clamp zero-sized chunk edges requested by `kerchunk.hdf4` to 1.

    HDF4 granules routinely hold zero-length variables — a MODIS fire-mask granule
    that detected no fires stores every one of its 27 `FP_*` fire-pixel variables
    with shape `(0,)`. `HDF4ToZarr.translate` passes those zero-length dimensions
    through as the chunk shape (`chunks=v.get("chunks", v["dims"])`), which zarr
    rejects from 3.3.0 onwards because a chunk edge must be positive. A zero-length
    array holds no chunks whatever its chunk shape, so coercing the edge to 1 is
    lossless, and it matches what `from_kerchunk_refs` already does when reading
    such references back.

    TODO: Remove once kerchunk clamps the chunk shape itself.
    """
    import zarr

    with _patch_lock:
        original = zarr.Group.require_array

        def require_array(self: zarr.Group, name: str, **kwargs: Any) -> Any:
            chunks = kwargs.get("chunks")
            if isinstance(chunks, Iterable):
                kwargs["chunks"] = tuple(edge or 1 for edge in chunks)
            return original(self, name, **kwargs)

        zarr.Group.require_array = require_array  # type: ignore[method-assign]
        try:
            yield
        finally:
            zarr.Group.require_array = original  # type: ignore[method-assign]


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
        with _positive_chunk_edges():
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
