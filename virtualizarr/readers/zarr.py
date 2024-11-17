import io

import zarr
from xarray import Variable

from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.zarr import ZArray


def virtual_variable_from_zarr_array(za: zarr.Array) -> Variable:
    """
    Create a virtual xarray.Variable wrapping a ManifestArray from a single zarr.Array.
    """

    # TODO this only works with zarr-python v2 for now

    attrs = dict(za.attrs)

    # extract _ARRAY_DIMENSIONS and remove it from attrs
    # TODO handle v3 DIMENSION_NAMES too
    dims = attrs.pop("_ARRAY_DIMENSIONS")

    zarray = ZArray(
        shape=za.shape,
        chunks=za.chunks,
        dtype=za.dtype,
        fill_value=za.fill_value,
        order=za.order,
        compressor=za.compressor,
        filters=za.filters,
        # zarr_format=za.zarr_format,
    )

    manifest = chunkmanifest_from_zarr_array(za)

    ma = ManifestArray(chunkmanifest=manifest, zarray=zarray)

    return Variable(data=ma, dims=dims, attrs=attrs)


def chunkmanifest_from_zarr_array(za: zarr.Array) -> ChunkManifest:
    import ujson

    of2 = io.StringIO()

    # TODO handle remote urls
    za.store.write_fsspec(of2)  # , url=url)
    out = ujson.loads(of2.getvalue())

    print(out)
