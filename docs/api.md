# API reference

VirtualiZarr has a small API surface, because most of the complexity is handled by xarray functions like ``xarray.concat`` and ``xarray.merge``.
Users can use xarray for every step apart from reading and serializing virtual references.

## User API

### Reading

::: virtualizarr.open_virtual_dataset

### Parsers

Each parser understands how to read a specific file format, and a parser must be passed to `virtualizarr.open_virtual_dataset`.

::: virtualizarr.parsers.DMRPPParser
::: virtualizarr.parsers.FITSParser
::: virtualizarr.parsers.HDFParser
::: virtualizarr.parsers.NetCDF3Parser
::: virtualizarr.parsers.KerchunkJSONParser
::: virtualizarr.parsers.KerchunkParquetParser
::: virtualizarr.parsers.ZarrParser

### Serialization

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor

::: virtualizarr.accessor.VirtualiZarrDataTreeAccessor

### Information

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.nbytes

### Rewriting
---------

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.rename_paths

### Developer API

If you want to write a new parser to create virtual references pointing to a custom file format, you will need to use VirtualiZarr's internal classes.
See the page on custom parsers for more information.

#### Manifests

VirtualiZarr uses these classes to store virtual references internally.
See the page on data structures for more information.

::: virtualizarr.manifests.ChunkManifest
::: virtualizarr.manifests.ManifestArray
::: virtualizarr.manifests.ManifestGroup
::: virtualizarr.manifests.ManifestStore

#### Array API

VirtualiZarr's [virtualizarr.ManifestArray][] objects support a limited subset of the Python Array API standard in `virtualizarr.manifests.array_api`.

::: virtualizarr.manifests.array_api.concatenate
::: virtualizarr.manifests.array_api.stack
::: virtualizarr.manifests.array_api.expand_dims
::: virtualizarr.manifests.array_api.broadcast_to

#### Parser typing protocol

All custom parsers must follow the `virtualizarr.parsers.typing.Parser` typing protocol.

::: virtualizarr.parsers.typing.Parser