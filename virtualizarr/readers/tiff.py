import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional

from xarray import Dataset, Index

from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
    open_loadable_vars_and_indexes,
)
from virtualizarr.translators.kerchunk import (
    extract_group,
    virtual_vars_and_metadata_from_kerchunk_refs,
)
from virtualizarr.types.kerchunk import KerchunkStoreRefs
from virtualizarr.utils import check_for_collisions


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

        drop_variables, loadable_variables = check_for_collisions(
            drop_variables=drop_variables, loadable_variables=loadable_variables
        )

        if reader_options is None:
            reader_options = {}

        reader_options.pop("storage_options", {})
        warnings.warn(
            "storage_options have been dropped from reader_options as they are not supported by kerchunk.tiff.tiff_to_zarr",
            UserWarning,
        )

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs({"refs": tiff_to_zarr(filepath, **reader_options)})

        refs = extract_group(refs, group)

        virtual_vars, attrs, coord_names = virtual_vars_and_metadata_from_kerchunk_refs(
            refs,
            loadable_variables,
            drop_variables,
            fs_root=Path.cwd().as_uri(),
        )

        loadable_vars, indexes = open_loadable_vars_and_indexes(
            filepath,
            loadable_variables=loadable_variables,
            reader_options=reader_options,
            drop_variables=drop_variables,
            indexes=indexes,
            group=group,
            decode_times=decode_times,
        )

        return construct_virtual_dataset(
            virtual_vars=virtual_vars,
            loadable_vars=loadable_vars,
            indexes=indexes,
            coord_names=coord_names,
            attrs=attrs,
        )
