from __future__ import annotations

import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr
from xarray.backends.zarr import FillValueCoder

from virtualizarr.codecs import numcodec_config_to_configurable
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.manifest import validate_and_normalize_path_to_uri
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.readers.api import VirtualBackend
from virtualizarr.readers.common import (
    construct_fully_virtual_dataset,
    replace_virtual_with_loadable_vars,
)
from virtualizarr.readers.hdf.filters import cfcodec_from_dataset, codecs_from_dataset
from virtualizarr.types import ChunkKey
from virtualizarr.utils import _FsspecFSFromFilepath, soft_import

h5py = soft_import("h5py", "For reading hdf files", strict=False)


if TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
    from h5py import Group as H5Group
    from obstore.store import ObjectStore

FillValueType = Union[
    int,
    float,
    bool,
    complex,
    str,
    np.integer,
    np.floating,
    np.bool_,
    np.complexfloating,
    bytes,  # For fixed-length string storage
    Tuple[bytes, int],  # Structured type
]


class HDFVirtualBackend(VirtualBackend):
    @staticmethod
    def _construct_manifest_array(
        path: str,
        dataset: H5Dataset,
        group: str,
    ) -> ManifestArray:
        """
        Construct a ManifestArray from an h5py dataset
        Parameters
        ----------
        path: str
            The path of the hdf5 file.
        dataset : h5py.Dataset
            An h5py dataset.
        group : str
            Name of the group containing this h5py.Dataset.
        Returns
        -------
        ManifestArray
        """
        chunks = dataset.chunks if dataset.chunks else dataset.shape
        codecs = codecs_from_dataset(dataset)
        attrs = HDFVirtualBackend._extract_attrs(dataset)
        dtype = dataset.dtype

        codec_configs = [
            numcodec_config_to_configurable(codec.get_config()) for codec in codecs
        ]

        fill_value = dataset.fillvalue.item()
        dims = tuple(HDFVirtualBackend._dataset_dims(dataset, group=group))
        metadata = create_v3_array_metadata(
            shape=dataset.shape,
            data_type=dtype,
            chunk_shape=chunks,
            fill_value=fill_value,
            codecs=codec_configs,
            dimension_names=dims,
            attributes=attrs,
        )

        manifest = HDFVirtualBackend._dataset_chunk_manifest(path, dataset)
        return ManifestArray(metadata=metadata, chunkmanifest=manifest)

    @staticmethod
    def _construct_manifest_group(
        store: ObjectStore,
        filepath: str,
        *,
        group: str | None = None,
        drop_variables: Optional[List[str]] = None,
    ) -> ManifestGroup:
        """
        Construct a virtual Group from a HDF dataset.
        """
        from virtualizarr.utils import ObstoreReader

        if drop_variables is None:
            drop_variables = []

        reader = ObstoreReader(store=store, path=filepath)
        f = h5py.File(reader, mode="r")

        if group is not None and group != "":
            g = f[group]
            group_name = group
            if not isinstance(g, h5py.Group):
                raise ValueError("The provided group is not an HDF group")
        else:
            g = f["/"]
            group_name = "/"

        manifest_dict = {}
        non_coordinate_dimesion_vars = HDFVirtualBackend._find_non_coord_dimension_vars(
            group=g
        )
        drop_variables = list(set(drop_variables + non_coordinate_dimesion_vars))
        attrs: dict[str, Any] = {}
        for key in g.keys():
            if key not in drop_variables:
                if isinstance(g[key], h5py.Dataset):
                    variable = HDFVirtualBackend._construct_manifest_array(
                        path=filepath,
                        dataset=g[key],
                        group=group_name,
                    )
                    if variable is not None:
                        manifest_dict[key] = variable
        return ManifestGroup(arrays=manifest_dict, attributes=attrs)

    @staticmethod
    def _create_manifest_store(
        filepath: str,
        *,
        prefix: str,
        store: ObjectStore,
        group: str | None = None,
    ) -> ManifestStore:
        # Create a group containing dataset level metadata and all the manifest arrays
        manifest_group = HDFVirtualBackend._construct_manifest_group(
            store=store, filepath=filepath, group=group
        )
        # Convert to a manifest store
        return ManifestStore(stores={prefix: store}, group=manifest_group)

    @staticmethod
    def open_virtual_dataset(
        filepath: str,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
        loadable_variables: Iterable[str] | None = None,
        decode_times: bool | None = None,
        indexes: Mapping[str, xr.Index] | None = None,
        virtual_backend_kwargs: Optional[dict] = None,
        reader_options: Optional[dict] = None,
    ) -> xr.Dataset:
        if h5py is None:
            raise ImportError("h5py is required for using the HDFVirtualBackend")
        if virtual_backend_kwargs:
            raise NotImplementedError(
                "HDF reader does not understand any virtual_backend_kwargs"
            )

        filepath = validate_and_normalize_path_to_uri(
            filepath, fs_root=Path.cwd().as_uri()
        )

        _drop_vars: list[Hashable] = (
            [] if drop_variables is None else list(drop_variables)
        )

        # TODO provide a way to drop a variable _before_ h5py attempts to inspect it?
        virtual_vars = HDFVirtualBackend._virtual_vars_from_hdf(
            path=filepath,
            group=group,
            reader_options=reader_options,
        )

        attrs = HDFVirtualBackend._get_group_attrs(
            path=filepath, reader_options=reader_options, group=group
        )
        coordinates_attr = attrs.pop("coordinates", "")
        coord_names = coordinates_attr.split()

        fully_virtual_dataset = construct_fully_virtual_dataset(
            virtual_vars=virtual_vars,
            coord_names=coord_names,
            attrs=attrs,
        )

        vds = replace_virtual_with_loadable_vars(
            fully_virtual_dataset,
            filepath,
            group=group,
            loadable_variables=loadable_variables,
            reader_options=reader_options,
            indexes=indexes,
            decode_times=decode_times,
        )

        return vds.drop_vars(_drop_vars)

    @staticmethod
    def _dataset_chunk_manifest(
        path: str,
        dataset: H5Dataset,
    ) -> ChunkManifest:
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
                chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
            else:
                key_list = [0] * (len(dataset.shape) or 1)
                key = ".".join(map(str, key_list))

                chunk_entry: ChunkEntry = ChunkEntry.with_validation(  # type: ignore[attr-defined]
                    path=path, offset=dsid.get_offset(), length=dsid.get_storage_size()
                )
                chunk_key = ChunkKey(key)
                chunk_entries = {chunk_key: chunk_entry}
                chunk_manifest = ChunkManifest(entries=chunk_entries)
        else:
            num_chunks = dsid.get_num_chunks()
            if num_chunks == 0:
                chunk_manifest = ChunkManifest(entries={}, shape=dataset.shape)
            else:
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
    def _dataset_dims(dataset: H5Dataset, group: str = "") -> List[str]:
        """
        Get a list of dimension scale names attached to input HDF5 dataset.

        This is required by the xarray package to work with Zarr arrays. Only
        one dimension scale per dataset dimension is allowed. If dataset is
        dimension scale, it will be considered as the dimension to itself.

        Parameters
        ----------
        dataset : h5py.Dataset
            An h5py dataset.
        group : str
            Name of the group we are pulling these dimensions from. Required for potentially removing subgroup prefixes.

        Returns
        -------
        list[str]
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

        if not group.endswith("/"):
            group += "/"

        return [dim.removeprefix(group) for dim in dims]

    @staticmethod
    def _extract_cf_fill_value(
        h5obj: Union[H5Dataset, H5Group],
    ) -> Optional[FillValueType]:
        """
        Convert the _FillValue attribute from an HDF5 group or dataset into
        encoding.

        Parameters
        ----------
        h5obj : h5py.Group or h5py.Dataset
            An h5py group or dataset.
        """
        fillvalue = None
        for n, v in h5obj.attrs.items():
            if n == "_FillValue":
                if isinstance(v, np.ndarray) and v.size == 1:
                    fillvalue = v.item()
                else:
                    fillvalue = v
                fillvalue = FillValueCoder.encode(fillvalue, h5obj.dtype)  # type: ignore[arg-type]
        return fillvalue

    @staticmethod
    def _extract_attrs(h5obj: Union[H5Dataset, H5Group]):
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
            if n == "_FillValue":
                continue
            # Fix some attribute values to avoid JSON encoding exceptions...
            if isinstance(v, bytes):
                v = v.decode("utf-8") or " "
            elif isinstance(v, (np.ndarray, np.number, np.bool_)):
                if v.dtype.kind == "S":
                    v = v.astype(str)
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
    def _dataset_to_variable(
        path: str,
        dataset: H5Dataset,
        group: str,
    ) -> Optional[xr.Variable]:
        """
        Extract an xarray Variable with ManifestArray data from an h5py dataset

        Parameters
        ----------
        dataset : h5py.Dataset
            An h5py dataset.
        group : str
            Name of the group containing this h5py.Dataset.

        Returns
        -------
        list: xarray.Variable
            A list of xarray variables.
        """
        chunks = dataset.chunks if dataset.chunks else dataset.shape
        codecs = codecs_from_dataset(dataset)
        cfcodec = cfcodec_from_dataset(dataset)
        attrs = HDFVirtualBackend._extract_attrs(dataset)
        cf_fill_value = HDFVirtualBackend._extract_cf_fill_value(dataset)
        attrs.pop("_FillValue", None)

        if cfcodec:
            codecs.insert(0, cfcodec["codec"])
            dtype = cfcodec["target_dtype"]
            attrs.pop("scale_factor", None)
            attrs.pop("add_offset", None)
        else:
            dtype = dataset.dtype

        codec_configs = [
            numcodec_config_to_configurable(codec.get_config()) for codec in codecs
        ]

        fill_value = dataset.fillvalue.item()
        metadata = create_v3_array_metadata(
            shape=dataset.shape,
            data_type=dtype,
            chunk_shape=chunks,
            fill_value=fill_value,
            codecs=codec_configs,
        )
        dims = HDFVirtualBackend._dataset_dims(dataset, group=group)
        manifest = HDFVirtualBackend._dataset_chunk_manifest(path, dataset)
        if manifest:
            marray = ManifestArray(metadata=metadata, chunkmanifest=manifest)
            variable = xr.Variable(data=marray, dims=dims, attrs=attrs)
        else:
            variable = xr.Variable(data=np.empty(dataset.shape), dims=dims, attrs=attrs)
        if cf_fill_value is not None:
            variable.encoding["_FillValue"] = cf_fill_value
        return variable

    @staticmethod
    def _virtual_vars_from_hdf(
        path: str,
        group: Optional[str] = None,
        drop_variables: Optional[List[str]] = None,
        reader_options: Optional[dict] = {
            "storage_options": {"key": "", "secret": "", "anon": True}
        },
    ) -> Dict[str, xr.Variable]:
        """
        Extract xarray Variables with ManifestArray data from an HDF file or group

        Parameters
        ----------
        path: str
            The path of the hdf5 file.
        group: str, optional
            The name of the group for which to extract variables. None refers to the root group.
        drop_variables: list of str
            A list of variable names to skip extracting.
        reader_options: dict
            A dictionary of reader options passed to fsspec when opening the file.

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

        if group is not None and group != "":
            g = f[group]
            group_name = group
            if not isinstance(g, h5py.Group):
                raise ValueError("The provided group is not an HDF group")
        else:
            g = f["/"]
            group_name = "/"

        variables = {}
        non_coordinate_dimesion_vars = HDFVirtualBackend._find_non_coord_dimension_vars(
            group=g
        )
        drop_variables = list(set(drop_variables + non_coordinate_dimesion_vars))
        for key in g.keys():
            if key not in drop_variables:
                if isinstance(g[key], h5py.Dataset):
                    variable = HDFVirtualBackend._dataset_to_variable(
                        path=path,
                        dataset=g[key],
                        group=group_name,
                    )
                    if variable is not None:
                        variables[key] = variable
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

    @staticmethod
    def _find_non_coord_dimension_vars(group: H5Group) -> List[str]:
        dimension_names = []
        non_coordinate_dimension_variables = []
        for name, obj in group.items():
            if "_Netcdf4Dimid" in obj.attrs:
                dimension_names.append(name)
        for name, obj in group.items():
            if type(obj) is h5py.Dataset:
                if obj.id.get_storage_size() == 0 and name in dimension_names:
                    non_coordinate_dimension_variables.append(name)

        return non_coordinate_dimension_variables
