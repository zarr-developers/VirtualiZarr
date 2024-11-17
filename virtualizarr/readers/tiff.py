from typing import Iterable, Mapping, Optional

import zarr
from xarray import DataArray, Dataset, Index

from virtualizarr.readers.common import VirtualBackend
from virtualizarr.readers.zarr import virtual_variable_from_zarr_array


class TIFFVirtualBackend(VirtualBackend):
    @staticmethod
    def open_virtual_dataarray(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        from tifffile import imread

        store = imread(filepath, aszarr=True)

        # TODO exception handling for TIFF files with multiple arrays
        za = zarr.open_array(store=store, mode="r")

        vv = virtual_variable_from_zarr_array(za)

        # TODO should we generate any pixel coordnate arrays like kerhunk seems to do?

        return DataArray(data=vv, dims=vv.dims, attrs=za.attrs)

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
        from tifffile import imread

        store = imread(filepath, aszarr=True)

        try:
            zg = zarr.open_group(store, mode="r")
        except zarr.errors.ContainsArrayError:
            # TODO tidy this up
            print(
                "TIFF file contains only a single array, please use `open_virtual_dataarray` instead"
            )
            raise

        raise NotImplementedError()
