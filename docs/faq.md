# FAQ

### How does this work?

I'm glad you asked! We can think of the problem of providing virtualized zarr-like access to a set of legacy files in some other format as a series of steps:

1) **Read byte ranges** - We use the various [kerchunk file format backends](https://fsspec.github.io/kerchunk/reference.html#file-format-backends) to determine which byte ranges within a given legacy file would have to be read in order to get a specific chunk of data we want.
2) **Construct a representation of a single file (or array within a file)** - Kerchunk's backends return a nested dictionary representing an entire file, but we instead immediately parse this dict and wrap it up into a set of `ManifestArray` objects. The record of where to look to find the file and the byte ranges is stored under the `ManifestArray.manifest` attribute, in a `ChunkManifest` object. Both steps (1) and (2) are handled by the `'virtualizarr'` xarray backend, which returns one `xarray.Dataset` object per file, each wrapping multiple `ManifestArray` instances (as opposed to e.g. numpy/dask arrays).
3) **Deduce the concatenation order** - The desired order of concatenation can either be inferred from the order in which the datasets are supplied (which is what `xr.combined_nested` assumes), or it can be read from the coordinate data in the files (which is what `xr.combine_by_coords` does). If the ordering information is not present as a coordinate (e.g. because it's in the filename), a pre-processing step might be required.
4) **Check that the desired concatenation is valid** - Whether called explicitly by the user or implicitly via `xr.combine_nested/combine_by_coords/open_mfdataset`, `xr.concat` is used to concatenate/stack the wrapped `ManifestArray` objects. When doing this xarray will spend time checking that the array objects and any coordinate indexes can be safely aligned and concatenated. Along with opening files, and loading coordinates in step (3), this is the main reason why `xr.open_mfdataset` can take a long time to return a dataset created from a large number of files.
5) **Combine into one big dataset** - `xr.concat` dispatches to the `concat/stack` methods of the underlying `ManifestArray` objects. These perform concatenation by merging their respective Chunk Manifests. Using xarray's `combine_*` methods means that we can handle multi-dimensional concatenations as well as merging many different variables.
6) **Serialize the combined result to disk** - The resultant `xr.Dataset` object wraps `ManifestArray` objects which contain the complete list of byte ranges for every chunk we might want to read. We now serialize this information to disk, either using the [kerchunk specification](https://fsspec.github.io/kerchunk/spec.html#version-1), or in future we plan to use [new Zarr extensions](https://github.com/zarr-developers/zarr-specs/issues/287) to write valid Zarr stores directly.
7) **Open the virtualized dataset from disk** - The virtualized zarr store can now be read from disk, skipping all the work we did above. Chunk reads from this store will be redirected to read the corresponding bytes in the original legacy files.

**Note:** Using the `kerchunk` library alone will perform a similar set of steps overall, but because (3), (4), (5), and (6) are all performed by the `kerchunk.combine.MultiZarrToZarr` function, and no internal abstractions are exposed, the design is much less modular, and the use cases are limited by kerchunk's API surface.

### How does VirtualiZarr compare to Kerchunk?

