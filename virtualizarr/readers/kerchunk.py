from typing import Iterable, Mapping, Optional

import ujson
from xarray import Dataset, Index

from virtualizarr.readers.common import VirtualBackend
from virtualizarr.translators.kerchunk import dataset_from_kerchunk_refs
from virtualizarr.types.kerchunk import (
    KerchunkStoreRefs,
)
from virtualizarr.utils import _FsspecFSFromFilepath, check_for_collisions


class KerchunkVirtualBackend(VirtualBackend):
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
        """Reads existing kerchunk references (in JSON or parquet) format."""

        if virtual_backend_kwargs is None:
            virtual_backend_kwargs = {}

        fs_root = virtual_backend_kwargs.pop("fs_root", None)

        if virtual_backend_kwargs:
            raise NotImplementedError(
                f"Kerchunk reader does not understand any of the virtual_backend_kwargs {virtual_backend_kwargs}"
            )

        if group:
            raise NotImplementedError()

        loadable_variables, drop_variables = check_for_collisions(
            drop_variables=drop_variables,
            loadable_variables=loadable_variables,
        )

        if loadable_variables or indexes or decode_times:
            raise NotImplementedError()

        fs = _FsspecFSFromFilepath(filepath=filepath, reader_options=reader_options)

        # The kerchunk .parquet storage format isn't actually a parquet, but a directory that contains named parquets for each group/variable.
        if fs.filepath.endswith(".parquet") and fs.fs.isfile(
            f"{fs.filepath}/.zmetadata"
        ):
            from fsspec.implementations.reference import LazyReferenceMapper

            lrm = LazyReferenceMapper(filepath, fs.fs)

            # build reference dict from KV pairs in LazyReferenceMapper
            # is there a better / more preformant way to extract this?
            array_refs = {k: lrm[k] for k in lrm.keys()}

            full_reference = {"refs": array_refs}

            vds = dataset_from_kerchunk_refs(
                KerchunkStoreRefs(full_reference), fs_root=fs_root
            )

        # JSON has no magic bytes, but the Kerchunk version 1 spec starts with 'version':
        # https://fsspec.github.io/kerchunk/spec.html
        elif fs.read_bytes(9).startswith(b'{"version'):
            with fs.open_file() as of:
                refs = ujson.load(of)

            vds = dataset_from_kerchunk_refs(KerchunkStoreRefs(refs), fs_root=fs_root)

        else:
            raise ValueError(
                "The input Kerchunk reference did not seem to be in Kerchunk's JSON or Parquet spec: https://fsspec.github.io/kerchunk/spec.html. If your Kerchunk generated references are saved in parquet format, make sure the file extension is `.parquet`. The Kerchunk format autodetection is quite flaky, so if your reference matches the Kerchunk spec feel free to open an issue: https://github.com/zarr-developers/VirtualiZarr/issues"
            )

        # TODO would be more efficient to drop these before converting them into ManifestArrays, i.e. drop them from the kerchunk refs dict
        return vds.drop_vars(drop_variables)
