from typing import Iterable, Mapping, Optional

from kerchunk.hdf import SingleHdf5ToZarr
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


class HDF5VirtualBackend(VirtualBackend):
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
        refs = SingleHdf5ToZarr(
            filepath, inline_threshold=0, **reader_options
        ).translate()

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
