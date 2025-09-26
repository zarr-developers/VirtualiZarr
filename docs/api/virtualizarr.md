# Virtual Datasets

VirtualiZarr has a small API surface, because most of the complexity is handled by xarray functions like [xarray.concat][] and [xarray.merge][].
Users can use xarray for every step apart from reading and serializing virtual references.

## Reading

::: virtualizarr.open_virtual_dataset

::: virtualizarr.open_virtual_mfdataset

## Information

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.nbytes

## Renaming paths

::: virtualizarr.accessor.VirtualiZarrDatasetAccessor.rename_paths
