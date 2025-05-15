### Roadmap to VirtualiZarr v2.0

Steps within each phase are independent. Moving on to the next phase requires completing all steps from the prior phase.
# Phase 1
- [ ] Define the V2 top-level API following preferred structure in [#400](https://github.com/zarr-developers/VirtualiZarr/issues/400)
    - [ ] Add the parser protocol from [the V2 design doc](./v2.md) to `protocols.py`.
    - [ ] Add the `open_virtual_dataset` function from the [V2 design doc](./v2.md) to `api.py`. (Closes [#553](https://github.com/zarr-developers/VirtualiZarr/issues/553), [#245](https://github.com/zarr-developers/VirtualiZarr/issues/245))
- [ ] Allow instantiating a `ManifestStore` using either an `ObjectStore` or a `StoreRegistry`.

# Phase 2 (Closes [#498](https://github.com/zarr-developers/VirtualiZarr/issues/498))
- [ ] Modify HDF tests to not be parameterized over the two readers, to allow separate updates
- [ ] Modify HDF5 parser to use the code from `api` and `protocols`
- [ ] Modify the Kerchunk parsers to use the code from `api` and `protocols`
- [ ] Modify the dmr++ parser to use the code from `api` and `protocols`

# Phase 3
- [ ] Remove `StoreRegistry` components from `ObjectStore` (closes [#561](https://github.com/zarr-developers/VirtualiZarr/issues/561), [#559](https://github.com/zarr-developers/VirtualiZarr/issues/559))
- [ ] Move `virtualizarr/backend.py:open_virtual_mfdataset` to `api.py` and update according to the [V2 design doc](./v2.md).
- [ ] Remove `virtualizarr/backend.py`

# Phase 4
- [ ] Rename `virtualize` accessor to `vz` (Closes [#241](https://github.com/zarr-developers/VirtualiZarr/issues/241))
- [ ] Refactor for dtypes changes and pin minimum Zarr version to 3.1.0

# Phase 5 (documentation)
- [ ] Create a migration guide
- [ ] Update [front-page usage](https://virtualizarr.readthedocs.io/en/latest/index.html#usage)
- [ ] Update [usage](https://virtualizarr.readthedocs.io/en/latest/usage.html)
- [ ] Update [examples](https://virtualizarr.readthedocs.io/en/latest/examples.html)
- [ ] Update [FAQ](https://virtualizarr.readthedocs.io/en/latest/faq.html)
- [ ] Check [data structure](https://virtualizarr.readthedocs.io/en/latest/data_structures.html)
- [ ] Check [custom readers](https://virtualizarr.readthedocs.io/en/latest/custom_readers.html)
- [ ] Update all usage of readers to parsers [closes #239](https://github.com/zarr-developers/VirtualiZarr/issues/239)

# Phase 6 (release)
- [ ] Switch `develop` to `main` and encourage people to try it out before a v2.0 release via the Earthmover channel.
- [ ] Create a v2.0 release
- [ ] Publish one or more blog-posts
- [ ] Give a Pangeo showcase in the fall

# Phase-independent restructuring (wish-list) (Closes [#400](https://github.com/zarr-developers/VirtualiZarr/issues/400))
- [ ] Move `virtualizarr/tests/` to `tests/`
- [ ] Move `virtualizarr/xarray.py` to `virtualizarr/_core/xarray.py`
- [ ] Move `virtualizarr/utils.py` to `virtualizarr/_core/utils.py`
- [ ] Move `virtualizarr/parallel.py` to `virtualizarr/_core/parallel.py`
- [ ] Move `virtualizarr/readers/` to `virtualizarr/parsers/`
- [ ] Move any private components from `virtualizarr/manifests/` to `virtualizarr/_core/manifests`
- [ ] Move `virtualizarr` to `src/virtualizarr`

# Phase-independent tasks (wish-list)
- [ ] Use `store.get_range_async()` rather than `obstore.get_range_async()` in `ManifestStore` to support any `ObjectReader` in the future
- [ ] Integrate virtual_tiff as the TIFF reader
- [ ] Remove file suffix check from manifest [#582](https://github.com/zarr-developers/VirtualiZarr/issues/582)
- [ ] Enable virtualizing files with colons [#593](https://github.com/zarr-developers/VirtualiZarr/issues/593)
- [ ] Filenames without suffix, e.g. in Zarr V2 stores, are rejected [#373](https://github.com/zarr-developers/VirtualiZarr/issues/373)
- [ ] Add `to_icechunk` on ManifestGroup [#591](https://github.com/zarr-developers/VirtualiZarr/pull/591)
- [ ] Avoid loading scalar values automatically [#270](https://github.com/zarr-developers/VirtualiZarr/issues/270)
- [ ] open_virtual_dataset returns some coordinates as data variables [#189](https://github.com/zarr-developers/VirtualiZarr/issues/189)
- [ ] Inconsistent fill_val between ZArray constructor and from_kerchunk_refs() [#287](https://github.com/zarr-developers/VirtualiZarr/issues/287)
- [ ] Manipulation of coordinages do not materialize to kerchunk refs  [#281](https://github.com/zarr-developers/VirtualiZarr/issues/281)
- [ ] Inlined CF time variables fail round tripping when compressed. [#280](https://github.com/zarr-developers/VirtualiZarr/issues/280)
- [ ] Coordinates lost with GPM-IMERG file [#342](https://github.com/zarr-developers/VirtualiZarr/issues/342)
- [ ] Xarray assertions can trigger loading [#354](https://github.com/zarr-developers/VirtualiZarr/issues/354)
- [ ] Issue opening kerchunk reference files [#381](https://github.com/zarr-developers/VirtualiZarr/issues/381)
- [ ] Concatenation of virtual datasets fails due to missing Chunk Manager [#382](https://github.com/zarr-developers/VirtualiZarr/issues/382)
- [ ] Add [architecture diagram](https://github.com/zarr-developers/VirtualiZarr/issues/225)
- [ ] Start a cookbook:
    - [ ] Add example for HDF5
    - [ ] Add simple example for Kerchunk readers [closes #488](https://github.com/zarr-developers/VirtualiZarr/issues/448)
    - [ ] Add a simple example for DMR++
    - [ ] Add a simple example of adding a time dimension based on filepath
    - [ ] Add a simple example of virtualizing to Zarr rather than Xarray dataset
    - [ ] Add a simple example over S3 with credentials
