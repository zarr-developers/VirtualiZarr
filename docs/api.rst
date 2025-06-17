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

.. currentmodule:: virtualizarr.xarray
.. autosummary::
    :nosignatures:
    :toctree: generated/

    open_virtual_dataset

Parsers
-------

Each parser understands how to read a specific file format, and one parser must be passed to :py:func:`~virtualizarr.open_virtual_dataset`

.. currentmodule:: virtualizarr.parsers
.. autosummary::
    :nosignatures:
    :toctree: generated/

    DMRPPParser
    FITSParser
    HDFParser
    NetCDF3Parser
    KerchunkJSONParser
    KerchunkParquetParser
    ZarrParser

Serialization
-------------

.. currentmodule:: virtualizarr.accessor
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.to_kerchunk
    VirtualiZarrDatasetAccessor.to_icechunk
    VirtualiZarrDataTreeAccessor.to_icechunk

Information
-----------

.. currentmodule:: virtualizarr.accessor
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.nbytes

Rewriting
---------

.. currentmodule:: virtualizarr.accessor
.. autosummary::
    :nosignatures:
    :toctree: generated/

    VirtualiZarrDatasetAccessor.rename_paths

Developer API
=============

If you want to write a new parser to create virtual references pointing to a custom file format, you will need to use VirtualiZarr's internal classes.
See the page on custom parsers for more information.

Manifests
---------

VirtualiZarr uses these classes to store virtual references internally.
See the page on data structures for more information.

.. currentmodule:: virtualizarr.manifests
.. autosummary::
    :nosignatures:
    :toctree: generated/

    ChunkManifest
    ManifestArray
    ManifestGroup
    ManifestStore

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

Parser typing protocol
----------------------

All custom parsers must follow the :py:class:`~virtualizarr.parsers.typing.Parser` typing protocol.

.. currentmodule:: virtualizarr.parsers.typing
.. autosummary::
    :nosignatures:
    :toctree: generated/

    Parser
