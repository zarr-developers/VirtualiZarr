import warnings
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Optional

from xarray import Dataset, Index

from virtualizarr.manifests import ManifestStore
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.types.kerchunk import KerchunkStoreRefs


class TIFFVirtualBackend(VirtualBackend):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        if virtual_backend_kwargs:
            raise NotImplementedError(
                "TIFF reader does not understand any virtual_backend_kwargs"
            )

        from kerchunk.tiff import tiff_to_zarr

        if reader_options is None:
            reader_options = {}

        reader_options.pop("storage_options", {})
        warnings.warn(
            "storage_options have been dropped from reader_options as they are not supported by kerchunk.tiff.tiff_to_zarr",
            UserWarning,
        )

        _drop_vars: list[Hashable] = (
            [] if drop_variables is None else list(drop_variables)
        )

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs({"refs": tiff_to_zarr(filepath, **reader_options)})

        manifeststore = ManifestStore.from_kerchunk_refs(
            refs,
            group=group,
            fs_root=Path.cwd().as_uri(),
        )

        vds = manifeststore.to_virtual_dataset(
            group=group,
            loadable_variables=loadable_variables,
            indexes=indexes,
        )

        return vds.drop_vars(_drop_vars)
