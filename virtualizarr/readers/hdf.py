from typing import List, Mapping, Optional, Union

import fsspec
import h5py
import numpy as np
import xarray as xr

from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray
from virtualizarr.readers.hdf_filters import codecs_from_dataset
from virtualizarr.types import ChunkKey
from virtualizarr.zarr import ZArray


def _dataset_chunk_manifest(path: str, dataset: h5py.Dataset) -> ChunkManifest:
    """
    Generate ChunkManifest for HDF5 dataset.

    Parameters
    ----------
    path: str
        The path the HDF5 container file
     dset : h5py.Dataset
        HDF5 dataset for which to create a ChunkManifest

    Returns
    -------
    ChunkManifest
        A Virtualizarr ChunkManifest
    """
    dsid = dataset.id

    if dataset.chunks is None:
        if dsid.get_offset() is None:
            raise ValueError("Dataset has no space allocated in the file")
        else:
            key_list = [0] * (len(dataset.shape) or 1)
            key = ".".join(map(str, key_list))
            chunk_entry = ChunkEntry(
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

        chunk_entries = dict()

        def get_key(blob):
            key_list = [a // b for a, b in zip(blob.chunk_offset, dataset.chunks)]
            key = ".".join(map(str, key_list))
            return key

        def store_chunk_entry(blob):
            chunk_entries[get_key(blob)] = ChunkEntry(
                path=path, offset=blob.byte_offset, length=blob.size
            )

        has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))
        if has_chunk_iter:
            dsid.chunk_iter(store_chunk_entry)
        else:
            for index in range(num_chunks):
                store_chunk_entry(dsid.get_chunk_info(index))

        chunk_manifest = ChunkManifest(entries=chunk_entries)
        return chunk_manifest


def _dataset_dims(dataset: h5py.Dataset) -> Union[List[str], List[None]]:
    """
    Get a list of dimension scale names attached to input HDF5 dataset.

    This is required by the xarray package to work with Zarr arrays. Only
    one dimension scale per dataset dimension is allowed. If dataset is
    dimension scale, it will be considered as the dimension to itself.

    Parameters
    ----------
    dataset : h5py.Dataset
        HDF5 dataset.

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
            num_scales = len(dataset.dims[n])
            if num_scales == 1:
                dims.append(dataset.dims[n][0].name[1:])
            elif h5py.h5ds.is_scale(dataset.id):
                dims.append(dataset.name[1:])
            elif num_scales > 1:
                raise ValueError(
                    f"{dataset.name}: {len(dataset.dims[n])} "
                    f"dimension scales attached to dimension #{n}"
                )
            elif num_scales == 0:
                # Some HDF5 files do not have dimension scales.
                # If this is the case, `num_scales` will be 0.
                # In this case, we mimic netCDF4 and assign phony dimension names.
                # See https://github.com/fsspec/kerchunk/issues/41
                dims.append(f"phony_dim_{n}")
    return dims


def _extract_attrs(h5obj: Union[h5py.Dataset, h5py.Group]):
    """
    Extract attributes from an HDF5 group or dataset.

    Parameters
    ----------
    h5obj : h5py.Group or h5py.Dataset
        An HDF5 group or dataset.
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


def _dataset_to_variable(path: str, dataset: h5py.Dataset) -> xr.Variable:
    # This chunk determination logic mirrors zarr-python's create
    # https://github.com/zarr-developers/zarr-python/blob/main/zarr/creation.py#L62-L66
    chunks = dataset.chunks if dataset.chunks else dataset.shape
    codecs = codecs_from_dataset(dataset)
    filters = [codec.get_config() for codec in codecs]
    zarray = ZArray(
        chunks=chunks,
        compressor=None,
        dtype=dataset.dtype,
        fill_value=dataset.fillvalue,
        filters=filters,
        order="C",
        shape=dataset.shape,
        zarr_format=2,
    )
    manifest = _dataset_chunk_manifest(path, dataset)
    marray = ManifestArray(zarray=zarray, chunkmanifest=manifest)
    dims = _dataset_dims(dataset)
    attrs = _extract_attrs(dataset)
    variable = xr.Variable(data=marray, dims=dims, attrs=attrs)
    return variable


def virtual_vars_from_hdf(
    path: str,
    drop_variables: Optional[List[str]] = None,
) -> Mapping[str, xr.Variable]:
    if drop_variables is None:
        drop_variables = []
    fs, file_path = fsspec.core.url_to_fs(path)
    open_file = fs.open(path, "rb")
    f = h5py.File(open_file, mode="r")
    variables = {}
    for key in f.keys():
        if key not in drop_variables:
            if isinstance(f[key], h5py.Dataset):
                variable = _dataset_to_variable(path, f[key])
                variables[key] = variable
            else:
                raise NotImplementedError("Nested groups are not yet supported")

    return variables


def attrs_from_root_group(path: str):
    fs, file_path = fsspec.core.url_to_fs(path)
    open_file = fs.open(path, "rb")
    f = h5py.File(open_file, mode="r")
    attrs = _extract_attrs(f)
    return attrs
