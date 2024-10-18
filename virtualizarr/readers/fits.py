from typing import Iterable, Mapping, Optional

from kerchunk.fits import process_file
from xarray import Dataset
from xarray.core.indexes import Index

from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
    open_loadable_vars_and_indexes,
)
from virtualizarr.translators.kerchunk import (
    extract_group,
    virtual_vars_and_metadata_from_kerchunk_refs,
)


class FITSVirtualBackend(VirtualBackend):
    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        # handle inconsistency in kerchunk, see GH issue https://github.com/zarr-developers/VirtualiZarr/issues/160
        refs = {"refs": process_file(filepath, **reader_options)}

        refs = extract_group(refs, group)

        virtual_vars, attrs, coord_names = virtual_vars_and_metadata_from_kerchunk_refs(
            refs,
            loadable_variables,
            drop_variables,
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
