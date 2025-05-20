from __future__ import annotations

import math
import os
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Union,
)

import numpy as np

from virtualizarr.codecs import numcodec_config_to_configurable
from virtualizarr.manifests import (
    ChunkEntry,
    ChunkManifest,
    ManifestArray,
    ManifestGroup,
    ManifestStore,
)
from virtualizarr.manifests.store import ObjectStoreRegistry
from virtualizarr.manifests.utils import create_v3_array_metadata
from virtualizarr.parsers.hdf.filters import codecs_from_dataset
from virtualizarr.parsers.utils import encode_cf_fill_value
from virtualizarr.types import ChunkKey
from virtualizarr.utils import ObstoreReader, soft_import

h5py = soft_import("h5py", "For reading hdf files", strict=False)


if TYPE_CHECKING:
    from h5py import Dataset as H5Dataset
    from h5py import Group as H5Group
    from obstore.store import ObjectStore


def _construct_manifest_array(
    filepath: str,
    dataset: H5Dataset,
    group: str,
) -> ManifestArray:
    """
    Construct a ManifestArray from an h5py dataset
    Parameters
    ----------
    filepath: str
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
    attrs = _extract_attrs(dataset)
    dtype = dataset.dtype

    # Temporarily disable use CF->Codecs - TODO re-enable in subsequent PR.
    # cfcodec = cfcodec_from_dataset(dataset)
    # if cfcodec:
    # codecs.insert(0, cfcodec["codec"])
    # dtype = cfcodec["target_dtype"]
    # attrs.pop("scale_factor", None)
    # attrs.pop("add_offset", None)
    # else:
    # dtype = dataset.dtype

    if "_FillValue" in attrs:
        encoded_cf_fill_value = encode_cf_fill_value(attrs["_FillValue"], dtype)
        attrs["_FillValue"] = encoded_cf_fill_value

    codec_configs = [
        numcodec_config_to_configurable(codec.get_config()) for codec in codecs
    ]

    fill_value = dataset.fillvalue.item()
    dims = tuple(_dataset_dims(dataset, group=group))
    metadata = create_v3_array_metadata(
        shape=dataset.shape,
        data_type=dtype,
        chunk_shape=chunks,
        fill_value=fill_value,
        codecs=codec_configs,
        dimension_names=dims,
        attributes=attrs,
    )
    manifest = _dataset_chunk_manifest(filepath, dataset)
    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _construct_manifest_group(
    filepath: str,
    reader: ObstoreReader,
    *,
    group: str | None = None,
    drop_variables: Optional[Iterable[str]] = None,
) -> ManifestGroup:
    """
    Construct a virtual Group from a HDF dataset.
    """

    if drop_variables is None:
        drop_variables = []

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
    # Several of our test fixtures which use xr.tutorial data have
    # non coord dimensions serialized using big endian dtypes which are not
    # yet supported in zarr-python v3.  We'll drop these variables for the
    # moment until big endian support is included upstream.)

    non_coordinate_dimension_vars = _find_non_coord_dimension_vars(group=g)
    drop_variables = list(set(list(drop_variables) + non_coordinate_dimension_vars))
    attrs = _extract_attrs(g)
    for key in g.keys():
        if key not in drop_variables:
            if isinstance(g[key], h5py.Dataset):
                variable = _construct_manifest_array(
                    filepath=filepath,
                    dataset=g[key],
                    group=group_name,
                )
                if variable is not None:
                    manifest_dict[key] = variable
    return ManifestGroup(arrays=manifest_dict, attributes=attrs)


class Parser:
    def __init__(
        self,
        group: str | None = None,
        drop_variables: Iterable[str] | None = None,
    ):
        self.group = group
        self.drop_variables = drop_variables

    def __call__(
        self,
        file_url: str,
        object_store: ObjectStore,
    ) -> ManifestStore:
        if h5py is None:
            raise ImportError("h5py is required for using the hdf parser")
        # Create a group containing dataset level metadata and all the manifest arrays

        filename = os.path.basename(file_url)
        reader = ObstoreReader(store=object_store, path=filename)
        manifest_group = _construct_manifest_group(
            filepath=file_url,
            reader=reader,
            group=self.group,
            drop_variables=self.drop_variables,
        )
        registry = ObjectStoreRegistry({file_url: object_store})
        # Convert to a manifest store
        return ManifestStore(store_registry=registry, group=manifest_group)


def _dataset_chunk_manifest(
    filepath: str,
    dataset: H5Dataset,
) -> ChunkManifest:
    """
    Generate ChunkManifest for HDF5 dataset.

    Parameters
    ----------
    filepath: str
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
                path=filepath, offset=dsid.get_offset(), length=dsid.get_storage_size()
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
                paths[key] = filepath
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
            v = v
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
