# API reference

VirtualiZarr has a small API surface, because most of the complexity is handled by xarray functions like ``xarray.concat`` and ``xarray.merge``.
Users can use xarray for every step apart from reading and serializing virtual references.

## User API

### Reading

::: virtualizarr.open_virtual_dataset
::: virtualizarr.open_virtual_mfdataset

### Serialization

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor
::: virtualizarr.accessor.VirtualiZarrDataTreeAccessor

### Information

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.nbytes

### Rewriting

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.rename_paths

### Developer API

If you want to write a new reader to create virtual references pointing to a custom file format, you will need to use VirtualiZarr's internal classes.

#### Manifests

VirtualiZarr uses these classes to store virtual references internally.

::: virtualizarr.manifests.ChunkManifest
::: virtualizarr.manifests.ManifestArray

#### Array API

VirtualiZarr's [virtualizarr.ManifestArray][] objects support a limited subset of the Python Array API standard in `virtualizarr.manifests.array_api`.

::: virtualizarr.manifests.array_api.concatenate
::: virtualizarr.manifests.array_api.stack
::: virtualizarr.manifests.array_api.expand_dims
::: virtualizarr.manifests.array_api.broadcast_to

#### Parallelization

Parallelizing virtual reference generation can be done using a number of parallel execution frameworks.
Advanced users may want to call one of these executors directly.
See the docs page on Scaling.

::: virtualizarr.parallel.SerialExecutor
::: virtualizarr.parallel.DaskDelayedExecutor
::: virtualizarr.parallel.LithopsExecutor
