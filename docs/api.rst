#############
API Reference
#############

.. currentmodule:: virtualizarr


Manifests
=========

.. currentmodule:: virtualizarr.manifests
.. autosummary::
    :nosignatures:
    :toctree: generated/

    ChunkManifest
    ManifestArray


Xarray
======

.. currentmodule:: virtualizarr.xarray
.. autosummary:: 
    :nosignatures:
    :toctree: generated/

    open_virtual_dataset
    VirtualiZarrDatasetAccessor.to_kerchunk
    VirtualiZarrDatasetAccessor.to_zarr


Array API
=========

VirtualiZarr's :py:class:`~virtualizarr.ManifestArray` objects support a limited subset of the Python Array API standard in :py:mod:`virtualizarr.manifests.array_api`.

.. currentmodule:: virtualizarr.manifests.array_api
.. autosummary:: 
    :nosignatures:
    :toctree: generated/

    concatenate
    stack
    expand_dims
    broadcast_to
