from pathlib import Path
from typing import Hashable, Iterable, Mapping, Optional

from xarray import Dataset, Index

from virtualizarr.readers.api import VirtualBackend
from virtualizarr.translators.kerchunk import (
    virtual_vars_and_metadata_from_kerchunk_refs,
)
from virtualizarr.xarray import (
    construct_fully_virtual_dataset,
    construct_virtual_dataset,
)


class NetCDF3VirtualBackend(VirtualBackend):
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
        from kerchunk.netCDF3 import NetCDF3ToZarr

        if virtual_backend_kwargs:
            raise NotImplementedError(
                "netcdf3 reader does not understand any virtual_backend_kwargs"
            )

        _drop_vars: list[Hashable] = (
            [] if drop_variables is None else list(drop_variables)
        )

        refs = NetCDF3ToZarr(filepath, inline_threshold=0, **reader_options).translate()

        # both group=None and group='' mean to read root group
        if group:
            raise ValueError(
                "group kwarg passed, but netCDF3 files can't have multiple groups!"
            )

        virtual_vars, attrs, coord_names = virtual_vars_and_metadata_from_kerchunk_refs(
            refs,
            fs_root=Path.cwd().as_uri(),
        )

        fully_virtual_dataset = construct_fully_virtual_dataset(
            virtual_vars=virtual_vars,
            coord_names=coord_names,
            attrs=attrs,
        )

        vds = construct_virtual_dataset(
            fully_virtual_ds=fully_virtual_dataset,
            filepath=filepath,
            group=group,
            loadable_variables=loadable_variables,
            reader_options=reader_options,
            indexes=indexes,
            decode_times=decode_times,
        )

        return vds.drop_vars(_drop_vars)
