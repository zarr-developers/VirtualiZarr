from typing import Iterable, Mapping, Optional

import ujson
from xarray import Dataset
from xarray.core.indexes import Index

from virtualizarr.readers.common import VirtualBackend
from virtualizarr.translators.kerchunk import dataset_from_kerchunk_refs
from virtualizarr.types.kerchunk import (
    KerchunkStoreRefs,
)
from virtualizarr.utils import _FsspecFSFromFilepath


class KerchunkVirtualBackend(VirtualBackend):
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, Index] | None = None,
        reader_options: Optional[dict] = None,
    ) -> Dataset:
        """Reads existing kerchunk references (in JSON or parquet) format."""

        fs = _FsspecFSFromFilepath(filepath=filepath, reader_options=reader_options)

        # The kerchunk .parquet storage format isn't actually a parquet, but a directory that contains named parquets for each group/variable.
        if fs.filepath.endswith("ref.parquet"):
            from fsspec.implementations.reference import LazyReferenceMapper

            lrm = LazyReferenceMapper(filepath, fs.fs)

            # build reference dict from KV pairs in LazyReferenceMapper
            # is there a better / more preformant way to extract this?
            array_refs = {k: lrm[k] for k in lrm.keys()}

            full_reference = {"refs": array_refs}

            return dataset_from_kerchunk_refs(KerchunkStoreRefs(full_reference))

        # JSON has no magic bytes, but the Kerchunk version 1 spec starts with 'version':
        # https://fsspec.github.io/kerchunk/spec.html
        elif fs.read_bytes(9).startswith(b'{"version'):
            with fs.open_file() as of:
                refs = ujson.load(of)

            return dataset_from_kerchunk_refs(KerchunkStoreRefs(refs))

        else:
            raise ValueError(
                "The input Kerchunk reference did not seem to be in Kerchunk's JSON or Parquet spec: https://fsspec.github.io/kerchunk/spec.html. The Kerchunk format autodetection is quite flaky, so if your reference matches the Kerchunk spec feel free to open an issue: https://github.com/zarr-developers/VirtualiZarr/issues"
            )
