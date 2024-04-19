from typing import List

import h5py
import xarray as xr

from virtualizarr.manifests import ChunkEntry, ChunkManifest, ManifestArray
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
                path=path,
                offset=dsid.get_offset(),
                length=dsid.get_storage_size()
            )
            chunk_entries = {key: chunk_entry}
            chunk_manifest = ChunkManifest(
                entries=chunk_entries
            )
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
                path=path,
                offset=blob.byte_offset,
                length=blob.size
            )

        has_chunk_iter = callable(getattr(dsid, "chunk_iter", None))
        if has_chunk_iter:
            dsid.chunk_iter(store_chunk_entry)
        else:
            for index in range(num_chunks):
                store_chunk_entry(dsid.get_chunk_info(index))

        chunk_manifest = ChunkManifest(
            entries=chunk_entries
        )
        return chunk_manifest

def _dataset_dims(dataset: h5py.Dataset) -> List[str]:
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


def _dataset_to_variable(path: str, dataset: h5py.Dataset) -> xr.Variable:
    # This chunk determination logic mirrors zarr-python's create
    # https://github.com/zarr-developers/zarr-python/blob/main/zarr/creation.py#L62-L66
    chunks = dataset.chunks if dataset.chunks else dataset.shape
    zarray = ZArray(
        chunks=chunks,
        compressor=dataset.compression,
        dtype=dataset.dtype,
        fill_value=dataset.fillvalue,
        filters=None,
        order="C",
        shape=dataset.shape,
        zarr_format=2,
    )
    manifest = _dataset_chunk_manifest(path, dataset)
    marray = ManifestArray(zarray=zarray, chunkmanifest=manifest)
    dims = _dataset_dims(dataset)
    variable = xr.Variable(data=marray, dims=dims)
    return variable
