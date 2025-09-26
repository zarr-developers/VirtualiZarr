# Developer API

If you want to write a new parser to create virtual references pointing to a custom file format, you will need to use VirtualiZarr's internal classes.
See the page on custom parsers for more information.

## Manifests

VirtualiZarr uses these classes to store virtual references internally.
See the page on data structures for more information.

::: virtualizarr.manifests.ChunkManifest
::: virtualizarr.manifests.ManifestArray
::: virtualizarr.manifests.ManifestGroup
::: virtualizarr.manifests.ManifestStore

## Registry

::: virtualizarr.registry.Url
[Urls][virtualizarr.registry.Url] should be parseable by [urllib.parse.urlparse][].
::: virtualizarr.registry.ObjectStoreRegistry

## Array API

VirtualiZarr's [virtualizarr.manifests.ManifestArray][] objects support a limited subset of the Python Array API standard in `virtualizarr.manifests.array_api`.

::: virtualizarr.manifests.array_api.concatenate
::: virtualizarr.manifests.array_api.stack
::: virtualizarr.manifests.array_api.expand_dims
::: virtualizarr.manifests.array_api.broadcast_to

## Parallelization

Parallelizing virtual reference generation can be done using a number of parallel execution frameworks.
Advanced users may want to call one of these executors directly.
See the docs page on Scaling.

::: virtualizarr.parallel.SerialExecutor
::: virtualizarr.parallel.DaskDelayedExecutor
::: virtualizarr.parallel.LithopsEagerFunctionExecutor
