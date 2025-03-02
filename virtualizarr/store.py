from collections.abc import AsyncIterator
from typing import Iterable

import xarray as xr
from fsspec.asyn import AsyncFileSystem
from zarr.abc.store import RangeByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.metadata import ArrayV3Metadata

from virtualizarr.manifests import ManifestArray


def validate_dataset_is_fully_virtual(vds: xr.Dataset) -> None:
    for name, var in vds.variables.items():
        if not isinstance(var.data, ManifestArray):
            raise TypeError(
                f"All variables in the dataset must be virtual, but variable {name} is of type {type(var.data)}, not type `ManifestArray`"
            )


class ManifestStore(Store):
    supports_writes: bool = False
    supports_deletes: bool = False
    supports_partial_writes: bool = False
    supports_listing: bool = True

    fs: AsyncFileSystem
    # TODO probably this should be dict[path, Dataset] to cover groups, but not enforce coordinate inheritance like xarray.DataTree would.
    vds: xr.Dataset

    def __init__(self, fs, vds, read_only=True):
        # TODO do we need to specify that this is v3 only?

        super().__init__(read_only=read_only)
        self._fs = fs

        validate_dataset_is_fully_virtual(vds)
        self._vds = vds

    def vds(self) -> xr.Dataset:
        """The Virtual Dataset whose byte references this store reads from."""
        return self._vds

    def fs(self) -> AsyncFileSystem:
        """The Fsspec FileSystem the byte ranges are defined for."""
        return self._fs

    async def clear(self) -> None:
        self.fs = None
        self.vds = None

    def __str__(self) -> str:
        return f"manifest://{id(self.vds)}"

    def __repr__(self) -> str:
        return f"ManifestStore('{self}')"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.fs == other.fs
            and xr.testing.assert_identical(self.vds, other.vds)
        )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: None = None,  # could this optionally accept a RangeByteRequest?
    ) -> Buffer | None:
        if not self._is_open:
            await self._open()

        if key.endswith("zarr.json"):
            array_path = key.removesuffix("zarr.json")
            metadata = self.get_array_metadata(array_path)
            # I don't understand why there is a single-element dict to unpack here...
            return metadata.to_buffer_dict["zarr.json"]

        print("key: ", key)
        print("key split: ", key.split("/"))
        array_name, _, chunk_key = key.split("/")
        # TODO: is this the best way?
        url, offset, length = self.vds[array_name].data.manifest.dict()[chunk_key]
        value = prototype.buffer.from_bytes(
            await self.fs._cat_file(
                url,
                start=offset,
                end=offset + length,
            )
        )
        return value

    def get_array_metadata(self, array_path: str) -> ArrayV3Metadata:
        *groups, array_name = array_path.split("/")

        if groups:
            # TODO support groups via dict[path, Dataset]
            raise NotImplementedError()
        else:
            var = self.vds.variables[array_name]
            return var.data.metadata

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, RangeByteRequest | None]],
    ) -> list[Buffer | None]:
        key_ranges = list(key_ranges)
        paths: list[str] = []
        starts: list[int] = []
        stops: list[int] = []
        for key, _ in key_ranges:
            array_name, _, chunk_key = key.split("/")
            url, offset, length = self.vds[array_name].data.manifest.dict()[chunk_key]
            paths.append(url)
            starts.append(offset)
            stops.append(offset + length)
        res = await self.fs._cat_ranges(paths, starts, stops, on_error="return")
        return [prototype.buffer.from_bytes(r) for r in res]

    async def exists(self, key: str) -> bool:
        array_name, _, chunk_key = key.split("/")
        url, _, _ = self.vds[array_name].data.manifest.dict()[chunk_key]
        return await self.fs._exists(url)

    async def list(self) -> AsyncIterator[str]:
        for array_name in self.vds.data_vars:
            for chunk_key in self.vds[array_name].data.manifest:
                yield f"{array_name}/{chunk_key}"

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def set(
        self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None
    ) -> None:
        raise NotImplementedError

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes]]
    ) -> None:
        raise NotImplementedError
