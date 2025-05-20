import os
from pathlib import Path
from typing import Iterable

import ujson
from obstore import open_reader
from obstore.store import ObjectStore

from virtualizarr.manifests import ManifestStore
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.translators.kerchunk import manifeststore_from_kerchunk_refs


class Parser:
    def __init__(
        self,
        group: str | None = None,
        fs_root: str | None = None,
        drop_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.fs_root = fs_root
        self.drop_variables = drop_variables

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        # TODO: whilst this keeps backwards-compatible behaviour for the `loadable_variables`` kwarg,
        # it probably has to change, see https://github.com/zarr-developers/VirtualiZarr/pull/477/#issuecomment-2744448626
        # if loadable_variables is None or indexes is None:
        # warnings.warn(
        # "The default value of the `loadable_variables` kwarg may attempt to load data from the referenced virtual chunks."
        # "As this is unlikely to be the desired behaviour when opening a Kerchunk file, `loadable_variables` has been overridden, and set to `loadable_variables=[]`."
        # "To silence this warning pass `loadable_variables` explicitly.",
        # UserWarning,
        # )
        # loadable_variables = []
        # indexes = {}
        filepath = validate_and_normalize_path_to_uri(
            file_url, fs_root=Path.cwd().as_uri()
        )
        filename = os.path.basename(filepath)
        reader = open_reader(store=object_store, path=filename)

        # The kerchunk .parquet storage format isn't actually a parquet, but a directory that contains named parquets for each group/variable.
        # if fs.filepath.endswith(".parquet") and fs.fs.isfile(
        # f"{fs.filepath}/.zmetadata"
        # ):
        # from fsspec.implementations.reference import LazyReferenceMapper

        # lrm = LazyReferenceMapper(filepath, fs.fs)

        # # build reference dict from KV pairs in LazyReferenceMapper
        # # is there a better / more performant way to extract this?
        # array_refs = {k: lrm[k] for k in lrm.keys()}

        # full_reference = {"refs": array_refs}

        # vds = dataset_from_kerchunk_refs(
        # KerchunkStoreRefs(full_reference), fs_root=fs_root
        # )

        # JSON has no magic bytes, but the Kerchunk version 1 spec starts with 'version':
        # https://fsspec.github.io/kerchunk/spec.html
        error_message = "The input Kerchunk reference did not seem to be in Kerchunk's JSON or Parquet spec: https://fsspec.github.io/kerchunk/spec.html. If your Kerchunk generated references are saved in parquet format, make sure the file extension is `.parquet`. The Kerchunk format autodetection is quite flaky, so if your reference matches the Kerchunk spec feel free to open an issue: https://github.com/zarr-developers/VirtualiZarr/issues"
        try:
            has_version = reader.read(9).to_bytes().startswith(b'{"version')
        except OSError:
            raise ValueError(error_message)
        if has_version:
            reader.seek(0)
            content = reader.read().to_bytes().decode()
            refs = ujson.loads(content)

            manifeststore = manifeststore_from_kerchunk_refs(
                refs,
                group=self.group,
                fs_root=self.fs_root,
                drop_variables=self.drop_variables,
            )
            return manifeststore
        else:
            raise ValueError(error_message)
