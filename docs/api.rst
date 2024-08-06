#############
API Reference
#############

.. currentmodule:: virtualizarr

VirtualiZarr has a small API surface, because most of the complexity is handled by xarray functions like ``xarray.concat`` and ``xarray.merge``.

Manifests
=========

.. currentmodule:: virtualizarr.manifests
.. autosummary::
    :nosignatures:
    :toctree: generated/

    ChunkManifest
    ManifestArray


Reading
=======

.. currentmodule:: virtualizarr.xarray
.. autosummary::
    :nosignatures:
    :toctree: generated/

    open_virtual_dataset


Serialization
=============

.. currentmodule:: virtualizarr.xarray
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.to_kerchunk
    VirtualiZarrDatasetAccessor.to_zarr


Rewriting
=============

.. currentmodule:: virtualizarr.xarray
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.rename_paths


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

Readers
=======

.. currentmodule:: virtualizarr.readers.dmrpp
.. autosummary::
    :nosignatures:
    :toctree: generated/

    DMRParser
    DMRParser.parse_dataset
