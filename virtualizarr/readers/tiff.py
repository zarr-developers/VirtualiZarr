import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional

from xarray import Dataset, Index

from virtualizarr.readers.api import VirtualBackend
from virtualizarr.readers.common import (
    construct_fully_virtual_dataset,
    replace_virtual_with_loadable_vars,
)
from virtualizarr.translators.kerchunk import (
    extract_group,
    virtual_vars_and_metadata_from_kerchunk_refs,
)
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

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs({"refs": tiff_to_zarr(filepath, **reader_options)})

        # both group=None and group='' mean to read root group
        if group:
            refs = extract_group(refs, group)

        virtual_vars, attrs, coord_names = virtual_vars_and_metadata_from_kerchunk_refs(
            refs,
            loadable_variables,
            drop_variables,
            fs_root=Path.cwd().as_uri(),
        )

        fully_virtual_dataset = construct_fully_virtual_dataset(
            virtual_vars=virtual_vars,
            coord_names=coord_names,
            attrs=attrs,
        )

        vds = replace_virtual_with_loadable_vars(
            fully_virtual_dataset,
            filepath,
            group=group,
            loadable_variables=loadable_variables,
            reader_options=reader_options,
            # drop_variables=drop_variables,
            indexes=indexes,
            decode_times=decode_times,
        )

        return vds.drop_vars(drop_variables)
