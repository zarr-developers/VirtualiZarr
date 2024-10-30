#############
API Reference
#############

.. currentmodule:: virtualizarr

VirtualiZarr has a small API surface, because most of the complexity is handled by xarray functions like ``xarray.concat`` and ``xarray.merge``.
Users can use xarray for every step apart from reading and serializing virtual references.

User API
========

Reading
-------

.. currentmodule:: virtualizarr.backend
.. autosummary::
    :nosignatures:
    :toctree: generated/

    open_virtual_dataset


Serialization
-------------

.. currentmodule:: virtualizarr.accessor
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.to_kerchunk
    VirtualiZarrDatasetAccessor.to_zarr
    VirtualiZarrDatasetAccessor.to_icechunk

Rewriting
---------

.. currentmodule:: virtualizarr.accessor
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.rename_paths

Developer API
=============

If you want to write a new reader to create virtual references pointing to a custom file format, you will need to use VirtualiZarr's internal classes.

Manifests
---------

VirtualiZarr uses these classes to store virtual references internally.

.. currentmodule:: virtualizarr.manifests
.. autosummary::
    :nosignatures:
    :toctree: generated/

    ChunkManifest
    ManifestArray


Array API
---------

VirtualiZarr's :py:class:`~virtualizarr.ManifestArray` objects support a limited subset of the Python Array API standard in :py:mod:`virtualizarr.manifests.array_api`.

.. currentmodule:: virtualizarr.manifests.array_api
.. autosummary::
    :nosignatures:
    :toctree: generated/

    concatenate
    stack
    expand_dims
    broadcast_to
