from pathlib import Path
from typing import Iterable, Mapping, Optional

from xarray import Dataset, Index, Variable

from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
)
from virtualizarr.translators.kerchunk import (
    extract_group,
    virtual_vars_and_metadata_from_kerchunk_refs,
)
from virtualizarr.types.kerchunk import KerchunkStoreRefs


class FITSVirtualBackend(VirtualBackend):
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
        from kerchunk.fits import process_file

        if virtual_backend_kwargs:
            raise NotImplementedError(
                "FITS reader does not understand any virtual_backend_kwargs"
            )

        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = KerchunkStoreRefs({"refs": process_file(filepath, **reader_options)})

        # both group=None and group='' mean to read root group
        if group:
            refs = extract_group(refs, group)

        # TODO This wouldn't work until either you had an xarray backend for FITS installed, or issue #124 is implemented to load data from ManifestArrays directly
        # TODO Once we have one of those we can use ``maybe_open_loadable_vars_and_indexes`` here
        if loadable_variables or indexes:
            raise NotImplementedError(
                "Cannot load variables or indexes from FITS files as there is no xarray backend engine for FITS"
            )
        loadable_vars: dict[str, Variable] = {}
        indexes = {}

        virtual_vars, attrs, coord_names = virtual_vars_and_metadata_from_kerchunk_refs(
            refs,
            loadable_variables,
            drop_variables,
            fs_root=Path.cwd().as_uri(),
        )

        return construct_virtual_dataset(
            virtual_vars=virtual_vars,
            loadable_vars=loadable_vars,
            indexes=indexes,
            coord_names=coord_names,
            attrs=attrs,
        )
