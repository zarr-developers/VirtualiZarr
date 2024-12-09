import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import xarray as xr
from xarray import Dataset, Index, Variable

from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
)
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.readers.common import (
    VirtualBackend,
    construct_virtual_dataset,
    open_loadable_vars_and_indexes,
)
from virtualizarr.readers.hdf.filters import cfcodec_from_dataset, codecs_from_dataset
from virtualizarr.types import ChunkKey
from virtualizarr.utils import _FsspecFSFromFilepath, check_for_collisions, soft_import
from virtualizarr.zarr import ZArray

if TYPE_CHECKING:
    import h5py  # type: ignore
    from h5py import Dataset, Group  # type: ignore

h5py = soft_import("h5py", "For reading hdf files", strict=False)
if h5py:
    Dataset = h5py.Dataset  # type: ignore
    Group = h5py.Group  # type: ignore
else:
    Dataset = dict()  # type: ignore
    Group = dict()  # type: ignore


class HDFVirtualBackend(VirtualBackend):
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
    ) -> xr.Dataset:
        if virtual_backend_kwargs:
            raise NotImplementedError(
                "HDF reader does not understand any virtual_backend_kwargs"
            )

        drop_variables, loadable_variables = check_for_collisions(
            drop_variables,
            loadable_variables,
        )

        filepath = validate_and_normalize_path_to_uri(
            filepath, fs_root=Path.cwd().as_uri()
        )

        virtual_vars = HDFVirtualBackend._virtual_vars_from_hdf(
            path=filepath,
            group=group,
            drop_variables=drop_variables + loadable_variables,
            reader_options=reader_options,
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

        attrs = HDFVirtualBackend._get_group_attrs(
            path=filepath, reader_options=reader_options, group=group
        )
        coordinates_attr = attrs.pop("coordinates", "")
        coord_names = coordinates_attr.split()
        return construct_virtual_dataset(
            virtual_vars=virtual_vars,
            loadable_vars=loadable_vars,
            indexes=indexes,
            coord_names=coord_names,
            attrs=attrs,
        )

    @staticmethod
    def _dataset_chunk_manifest(path: str, dataset: Dataset) -> Optional[ChunkManifest]:
        """
        Generate ChunkManifest for HDF5 dataset.

        Parameters
        ----------
        path: str
            The path of the HDF5 file
        dataset : h5py.Dataset
            h5py dataset for which to create a ChunkManifest

        Returns
        -------
        ChunkManifest
            A Virtualizarr ChunkManifest
        """
        dsid = dataset.id
        if dataset.chunks is None:
            if dsid.get_offset() is None:
                return None
            else:
                key_list = [0] * (len(dataset.shape) or 1)
                key = ".".join(map(str, key_list))

                chunk_entry = ChunkEntry.with_validation(
                    path=path, offset=dsid.get_offset(), length=dsid.get_storage_size()
                )
                chunk_key = ChunkKey(key)
                chunk_entries = {chunk_key: chunk_entry}
                chunk_manifest = ChunkManifest(entries=chunk_entries)
                return chunk_manifest
        else:
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                raise ValueError("The dataset is chunked but contains no chunks")
            shape = tuple(
                math.ceil(a / b) for a, b in zip(dataset.shape, dataset.chunks)
            )
            paths = np.empty(shape, dtype=np.dtypes.StringDType)  # type: ignore
            offsets = np.empty(shape, dtype=np.uint64)
            lengths = np.empty(shape, dtype=np.uint64)

            def get_key(blob):
                return tuple(
                    [a // b for a, b in zip(blob.chunk_offset, dataset.chunks)]
                )

            def add_chunk_info(blob):
                key = get_key(blob)
                paths[key] = path
                offsets[key] = blob.byte_offset
                lengths[key] = blob.size

            has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))
            if has_chunk_iter:
                dsid.chunk_iter(add_chunk_info)
            else:
                for index in range(num_chunks):
                    add_chunk_info(dsid.get_chunk_info(index))

            chunk_manifest = ChunkManifest.from_arrays(
                paths=paths,  # type: ignore
                offsets=offsets,
                lengths=lengths,
            )
            return chunk_manifest

    @staticmethod
    def _dataset_dims(dataset: Dataset) -> Union[List[str], List[None]]:
        """
        Get a list of dimension scale names attached to input HDF5 dataset.

        This is required by the xarray package to work with Zarr arrays. Only
        one dimension scale per dataset dimension is allowed. If dataset is
        dimension scale, it will be considered as the dimension to itself.

        Parameters
        ----------
        dataset : h5py.Dataset
            An h5py dataset.

        Returns
        -------
        list
            List with HDF5 path names of dimension scales attached to input
            dataset.
        """
        dims = list()
        rank = len(dataset.shape)
        if rank:
            for n in range(rank):
                num_scales = len(dataset.dims[n])  # type: ignore
                if num_scales == 1:
                    dims.append(dataset.dims[n][0].name[1:])  # type: ignore
                elif h5py.h5ds.is_scale(dataset.id):
                    dims.append(dataset.name[1:])
                elif num_scales > 1:
                    raise ValueError(
                        f"{dataset.name}: {len(dataset.dims[n])} "  # type: ignore
                        f"dimension scales attached to dimension #{n}"
                    )
                elif num_scales == 0:
                    # Some HDF5 files do not have dimension scales.
                    # If this is the case, `num_scales` will be 0.
                    # In this case, we mimic netCDF4 and assign phony dimension names.
                    # See https://github.com/fsspec/kerchunk/issues/41
                    dims.append(f"phony_dim_{n}")
        return dims

    @staticmethod
    def _extract_attrs(h5obj: Union[Dataset, Group]):
        """
        Extract attributes from an HDF5 group or dataset.

        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An h5py group or dataset.
        """
        _HIDDEN_ATTRS = {
            "REFERENCE_LIST",
            "CLASS",
            "DIMENSION_LIST",
            "NAME",
            "_Netcdf4Dimid",
            "_Netcdf4Coordinates",
            "_nc3_strict",
            "_NCProperties",
        }
        attrs = {}
        for n, v in h5obj.attrs.items():
            if n in _HIDDEN_ATTRS:
                continue
            # Fix some attribute values to avoid JSON encoding exceptions...
            if isinstance(v, bytes):
                v = v.decode("utf-8") or " "
            elif isinstance(v, (np.ndarray, np.number, np.bool_)):
                if v.dtype.kind == "S":
                    v = v.astype(str)
                if n == "_FillValue":
                    continue
                elif v.size == 1:
                    v = v.flatten()[0]
                    if isinstance(v, (np.ndarray, np.number, np.bool_)):
                        v = v.tolist()
                else:
                    v = v.tolist()
            elif isinstance(v, h5py._hl.base.Empty):
                v = ""
            if v == "DIMENSION_SCALE":
                continue

            attrs[n] = v
        return attrs

    @staticmethod
    def _dataset_to_variable(path: str, dataset: Dataset) -> Optional[Variable]:
        """
        Extract an xarray Variable with ManifestArray data from an h5py dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            An h5py dataset.

        Returns
        -------
        list: xarray.Variable
            A list of xarray variables.

        """
        # This chunk determination logic mirrors zarr-python's create
        # https://github.com/zarr-developers/zarr-python/blob/main/zarr/creation.py#L62-L66

        chunks = dataset.chunks if dataset.chunks else dataset.shape
        codecs = codecs_from_dataset(dataset)
        cfcodec = cfcodec_from_dataset(dataset)
        attrs = HDFVirtualBackend._extract_attrs(dataset)
        if cfcodec:
            codecs.insert(0, cfcodec["codec"])
            dtype = cfcodec["target_dtype"]
            attrs.pop("scale_factor", None)
            attrs.pop("add_offset", None)
            fill_value = cfcodec["codec"].decode(dataset.fillvalue)
        else:
            dtype = dataset.dtype
            fill_value = dataset.fillvalue
        if isinstance(fill_value, np.ndarray):
            fill_value = fill_value[0]
        if np.isnan(fill_value):
            fill_value = float("nan")
        if isinstance(fill_value, np.generic):
            fill_value = fill_value.item()
        filters = [codec.get_config() for codec in codecs]
        zarray = ZArray(
            chunks=chunks,  # type: ignore
            compressor=None,
            dtype=dtype,
            fill_value=fill_value,
            filters=filters,
            order="C",
            shape=dataset.shape,
            zarr_format=2,
        )
        dims = HDFVirtualBackend._dataset_dims(dataset)
        manifest = HDFVirtualBackend._dataset_chunk_manifest(path, dataset)
        if manifest:
            marray = ManifestArray(zarray=zarray, chunkmanifest=manifest)
            variable = Variable(data=marray, dims=dims, attrs=attrs)
        else:
            variable = Variable(data=np.empty(dataset.shape), dims=dims, attrs=attrs)
        return variable

    @staticmethod
    def _virtual_vars_from_hdf(
        path: str,
        group: Optional[str] = None,
        drop_variables: Optional[List[str]] = None,
        reader_options: Optional[dict] = {
            "storage_options": {"key": "", "secret": "", "anon": True}
        },
    ) -> Dict[str, Variable]:
        """
        Extract xarray Variables with ManifestArray data from an HDF file or group

        Parameters
        ----------
        path: str
            The path of the hdf5 file.
        group: str
            The name of the group for which to extract variables.
        drop_variables: list of str
            A list of variable names to skip extracting.
        reader_options: dict
            A dictionary of reader options passed to fsspec when opening the
            file.

        Returns
        -------
        dict
            A dictionary of Xarray Variables with the variable names as keys.

        """
        if drop_variables is None:
            drop_variables = []
        open_file = _FsspecFSFromFilepath(
            filepath=path, reader_options=reader_options
        ).open_file()
        f = h5py.File(open_file, mode="r")
        if group:
            g = f[group]
            if not isinstance(g, h5py.Group):
                raise ValueError("The provided group is not an HDF group")
        else:
            g = f
        variables = {}
        for key in g.keys():
            if key not in drop_variables:
                if isinstance(g[key], Dataset):
                    variable = HDFVirtualBackend._dataset_to_variable(path, g[key])
                    if variable is not None:
                        variables[key] = variable
                else:
                    raise NotImplementedError("Nested groups are not yet supported")

        return variables

    @staticmethod
    def _get_group_attrs(
        path: str,
        group: Optional[str] = None,
        reader_options: Optional[dict] = {
            "storage_options": {"key": "", "secret": "", "anon": True}
        },
    ):
        open_file = _FsspecFSFromFilepath(
            filepath=path, reader_options=reader_options
        ).open_file()
        f = h5py.File(open_file, mode="r")
        if group:
            g = f[group]
            if not isinstance(g, h5py.Group):
                raise ValueError("The provided group is not an HDF group")
        else:
            g = f
        attrs = HDFVirtualBackend._extract_attrs(g)
        return attrs
