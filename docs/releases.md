# Release notes

## v2.1.2 (3rd September 2025)

Patch release with minor bug fixes for the DMRPParser and Icechunk writing behavior.

### Bug fixes

- Enable `DMRPParser` to process scalar, dimensionless variables that lack chunks are present.
  ([#666](https://github.com/zarr-developers/VirtualiZarr/pull/757)).
  By [Miguel Jimenez-Urias](https://github.com/Mikejmnez).
- Enable `DMRPParser` to parse flattened dmrpp metadata reference files, which contain container attributes.
  ([#581](https://github.com/zarr-developers/VirtualiZarr/pull/757)).
  By [Miguel Jimenez-Urias](https://github.com/Mikejmnez).
- Support dtypes without an endianness ([#787](https://github.com/zarr-developers/VirtualiZarr/pull/787)). By [Justus Magin](https://github.com/keewis).

### Internal changes
- Change default Icechunk writing behavior to not validate or write "empty" chunks ([#791](https://github.com/zarr-developers/VirtualiZarr/pull/791)). By [Sean Harkins](https://github.com/sharkinsspatial).

## v2.1.1 (14th August 2025)

Extremely minor release to ensure compatibility with the soon-to-be released version of xarray (likely named v2025.07.2).

### Bug fixes

- Adjust for minor upcoming change in private xarray API `xarray.structure.combine._nested_combine`.
  ([#779](https://github.com/zarr-developers/VirtualiZarr/pull/779)).
  By [Tom Nicholas](https://github.com/TomNicholas).

## v2.1.0 (14th August 2025)

This release fixes a number of important bugs that could silently lead to referenced data being read back incorrectly.
In particular, note that writing virtual chunks to Icechunk now requires that all virtual chunk containers are set correctly by default.
It also unpins our dependency on xarray, so that VirtualiZarr is compatible with the latest released version of Xarray.
Please upgrade!

### New Features

- Expose `validate_containers` kwarg in `.to_icechunk`, allowing it to be set to `False` ([#567](https://github.com/zarr-developers/VirtualiZarr/pull/567), [#774](https://github.com/zarr-developers/VirtualiZarr/pull/774)).
  By [Tom Nicholas](https://github.com/TomNicholas).

### Breaking changes

- Writing to Icechunk now requires that virtual chunk containers are set correctly for all virtual references by default.
  ([#774](https://github.com/zarr-developers/VirtualiZarr/pull/774)).
  This change is needed because otherwise it can lead to situations in which attempting to read data back returns fill values instead of real data, silently! (See [#763](https://github.com/zarr-developers/VirtualiZarr/pull/763))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Update minimum required version of Icechunk to `v1.1.2` [#774](https://github.com/zarr-developers/VirtualiZarr/pull/774). By [Tom Nicholas](https://github.com/TomNicholas).
- Unpin dependency on xarray, by adjusting our tests to pass despite minor changes to the bytes of netCDF files written between versions of xarray [#774](https://github.com/zarr-developers/VirtualiZarr/pull/774)).
  By [Max Jones](https://github.com/maxrjones) and [Tom Nicholas](https://github.com/TomNicholas).

### Bug fixes

- Fixed bug where VirtualiZarr was incorrectly failing to raise if virtual chunk containers with correct prefixes were not set for every virtual reference ([#774](https://github.com/zarr-developers/VirtualiZarr/pull/774)).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Fix handling of big-endian data in Icechunk by making sure that non-default zarr serializers are included in the zarr array metadata [#766](https://github.com/zarr-developers/VirtualiZarr/issues/766).
  By [Max Jones](https://github.com/maxrjones)
- Fix handling of big-endian data in Kerchunk references [#769](https://github.com/zarr-developers/VirtualiZarr/issues/769).
  By [Max Jones](https://github.com/maxrjones)

### Documentation

- Updated Icechunk examples now that virtual chunk containers are required by default ([#774](https://github.com/zarr-developers/VirtualiZarr/pull/774)).
  By [Tom Nicholas](https://github.com/TomNicholas).

### Internal changes

- `extract_codecs` function inside `convert_to_codec_pipeline` now raises if it encounters a codec which does not inherit from the correct `zarr.abc.codec` base classes. ([#775](https://github.com/zarr-developers/VirtualiZarr/pull/775)).
  By [Tom Nicholas](https://github.com/TomNicholas).

## v2.0.1 (30th July 2025)

Minor release to ensure compatibility with incoming changes to Icechunk.

### Bug fixes

- Fixed bug caused by writing empty virtual chunks to Icechunk ([#745](https://github.com/zarr-developers/VirtualiZarr/pull/745)).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Rewrote the internals of `ManifestArray.__getitem__` to ensure it actually obeys the array API standard under myriad edge cases ([#734](https://github.com/zarr-developers/VirtualiZarr/pull/734)).
  By [Tom Nicholas](https://github.com/TomNicholas).

### Documentation

- Added recommendation to use `icechunk.Repository.save_config()` to persist `icechunk.VirtualChunkContainer`s ([#746](https://github.com/zarr-developers/VirtualiZarr/pull/746)).
  By [Tom Nicholas](https://github.com/TomNicholas).

## v2.0.0 (21st July 2025)

### New Features

- Added a pluggable system of "parsers" for generating virtual references from different filetypes. These follow the [`virtualizarr.parsers.typing.Parser`][] typing protocol, and return [`ManifestStore`][virtualizarr.manifests.ManifestStore] objects wrapping obstore stores.
  ([#498](https://github.com/zarr-developers/VirtualiZarr/issues/498), [#601](https://github.com/zarr-developers/VirtualiZarr/pull/601))
- Added a [Zarr parser][virtualizarr.parsers.ZarrParser] that allows opening Zarr V3 stores as virtual datasets.
  ([#271](https://github.com/zarr-developers/VirtualiZarr/pull/271)) By [Raphael Hagen](https://github.com/norlandrhagen).
- Added [`ManifestStore`][virtualizarr.manifests.ManifestStore] for loading data from ManifestArrays by ([#490](https://github.com/zarr-developers/VirtualiZarr/pull/490))
  By [Max Jones](https://github.com/maxrjones).
- Added [`ManifestStore.to_virtual_dataset()`][virtualizarr.manifests.ManifestStore.to_virtual_dataset] method ([#522](https://github.com/zarr-developers/VirtualiZarr/pull/522)).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Added [`open_virtual_mfdataset`][virtualizarr.open_virtual_mfdataset] function ([#345](https://github.com/zarr-developers/VirtualiZarr/issues/345), [#349](https://github.com/zarr-developers/VirtualiZarr/pull/349)).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Added `datatree_to_icechunk` function for writing an `xarray.DataTree` to
  an Icechunk store ([#244](https://github.com/zarr-developers/VirtualiZarr/issues/244)).  By [Chuck Daniels](https://github.com/chuckwondo).
- Added a `.vz` custom accessor to `xarray.DataTree`, exposing the method
  `xarray.DataTree.vz.to_icechunk()` for writing an `xarray.DataTree`
  to an Icechunk store ([#244](https://github.com/zarr-developers/VirtualiZarr/issues/244)).  By
  [Chuck Daniels](https://github.com/chuckwondo).
- Added a warning if you attempt to write an entirely non-virtual dataset to a virtual references format ([#657](https://github.com/zarr-developers/VirtualiZarr/pull/657)).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Support big-endian data via zarr-python 3.0.9 and zarr v3's new data types system ([#618](https://github.com/zarr-developers/VirtualiZarr/issues/618), [#677](https://github.com/zarr-developers/VirtualiZarr/issues/677)). By [Max Jones](https://github.com/maxrjones) and [Tom Nicholas](https://github.com/TomNicholas).
- Added a V1 -> V2 usage migration guide [#637](https://github.com/zarr-developers/VirtualiZarr/issues/637). By [Raphael Hagen](https://github.com/norlandrhagen).

### Breaking changes

- As [`virtualizarr.open_virtual_dataset`][] now uses parsers, it's API has changed. [#601](https://github.com/zarr-developers/VirtualiZarr/pull/601)) See the [migration-guide](migration_guide.md) for more details.
- The recommended virtualizarr Xarray accessor name is `vz` rather than `virtualize`.
- Which variables are loadable by default has changed. The behaviour is now to make loadable by default the
  same variables which `xarray.open_dataset` would create indexes for: i.e. one-dimensional coordinate variables whose
  name matches the name of their only dimension (also known as "dimension coordinates").
  Pandas indexes will also now be created by default for these loadable variables.
  This is intended to provide a more friendly default, as often you will want these small variables to be loaded
  (or "inlined", for efficiency of storage in icechunk/kerchunk), and you will also want to have in-memory indexes for these variables
  (to allow `xarray.combine_by_coords` to sort using them).
  The old behaviour is equivalent to passing `loadable_variables=[]` and `indexes={}`.
  ([#335](https://github.com/zarr-developers/VirtualiZarr/issues/335), [#477](https://github.com/zarr-developers/VirtualiZarr/pull/477)) by [Tom Nicholas](https://github.com/TomNicholas).
- Moved `ChunkManifest`, `ManifestArray` etc. to be behind a dedicated `.manifests` namespace. ([#620](https://github.com/zarr-developers/VirtualiZarr/issues/620), [#624](https://github.com/zarr-developers/VirtualiZarr/pull/624))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Now by default when writing virtual chunks to Icechunk, the `last_updated_time` for the chunk will be set to the current time. This helps protect users against reading from stale or overwritten chunks stored in Icechunk, by default.
  ([#436](https://github.com/zarr-developers/VirtualiZarr/issues/436), [#480](https://github.com/zarr-developers/VirtualiZarr/pull/480)) by [Tom Nicholas](https://github.com/TomNicholas).
- Minimum supported version of Icechunk is now `v1.0`
- Minimum supported version of Zarr is now `v3.1.0`
- Xarray is pinned to `v2025.6.0`. We expect to loosen the upper bound shortly.

### Bug fixes

- Fixed bug causing ManifestArrays to compare as not equal when they were actually identical ([#501](https://github.com/zarr-developers/VirtualiZarr/issues/501), [#502](https://github.com/zarr-developers/VirtualiZarr/pull/502))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Fixed bug causing coordinates to be demoted to data variables when writing to Icechunk ([#574](https://github.com/zarr-developers/VirtualiZarr/issues/574), [#588](https://github.com/zarr-developers/VirtualiZarr/pull/588))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Removed checks forbidding paths in virtual references without file suffixes ([#659](https://github.com/zarr-developers/VirtualiZarr/pull/659))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Fixed bug when indexing a scalar ManifestArray with an ellipsis([#596](https://github.com/zarr-developers/VirtualiZarr/issues/596), [#641](https://github.com/zarr-developers/VirtualiZarr/pull/641))
  By [Max Jones](https://github.com/maxrjones) and [Tom Nicholas](https://github.com/TomNicholas).

### Documentation

- Added more detail to error messages when an indexer of ManifestArray is invalid ([#630](https://github.com/zarr-developers/VirtualiZarr/issues/630), [#635](https://github.com/zarr-developers/VirtualiZarr/pull/635)). By [Danny Kaufman](https://github.com/danielfromearth/).
- Added new docs page on how to write a custom parser for bespoke file formats ([#452](https://github.com/zarr-developers/VirtualiZarr/issues/452), [#580](https://github.com/zarr-developers/VirtualiZarr/pull/580))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Added new docs page on how to scale VirtualiZarr effectively[#590](https://github.com/zarr-developers/VirtualiZarr/issues/590).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Documented the new [`virtualizarr.open_virtual_mfdataset`] function [#590](https://github.com/zarr-developers/VirtualiZarr/issues/590).
  By [Tom Nicholas](https://github.com/TomNicholas).
- Added MUR SST virtual and zarr icechunk store generation using lithops example.
  ([#475](https://github.com/zarr-developers/VirtualiZarr/pull/475)) by [Aimee Barciauskas](https://github.com/abarciauskas-bgse).
- Added FAQ answer about what data can be virtualized ([#430](https://github.com/zarr-developers/VirtualiZarr/issues/430), [#532](https://github.com/zarr-developers/VirtualiZarr/pull/532))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Switched docs build to use mkdocs-material instead of sphinx ([#615](https://github.com/zarr-developers/VirtualiZarr/pull/615))
  By [Max Jones](https://github.com/maxrjones).
- Moved examples into a `V1/` directory and adds notes that examples use the VirtualiZarr V1 syntax [#644](https://github.com/zarr-developers/VirtualiZarr/issues/644). By [Raphael Hagen](https://github.com/norlandrhagen).

### Internal Changes

- `ManifestArrays` now internally use [zarr.core.metadata.v3.ArrayV3Metadata](https://github.com/zarr-developers/zarr-python/blob/v3.0.2/src/zarr/core/metadata/v3.py). This replaces the `ZArray` class that was previously used to store metadata about manifest arrays. ([#429](https://github.com/zarr-developers/VirtualiZarr/pull/429)) By [Aimee Barciauskas](https://github.com/abarciauskas-bgse). Notable internal changes:
    - Make zarr-python a required dependency with a minimum version `>=3.0.2`.
    - Specify a minimum numcodecs version of `>=0.15.1`.
    - When creating a `ManifestArray`, the `metadata` property should be an `zarr.core.metadata.v3.ArrayV3Metadata` object. There is a helper function `create_v3_array_metadata` which should be used, as it has some useful defaults and includes `convert_to_codec_pipeline` (see next bullet).
    - The function `convert_to_codec_pipeline` ensures the codec pipeline passed to `ArrayV3Metadata` has valid codecs in the expected order (`ArrayArrayCodec`s, `ArrayBytesCodec`, `BytesBytesCodec`s) and includes the required `ArrayBytesCodec` using the default for the data type.
      - Note: `convert_to_codec_pipeline` uses the zarr-python function `get_codec_class` to convert codec configurations (i.e. `dict`s with a name and configuration key, see [parse_named_configuration](https://github.com/zarr-developers/zarr-python/blob/v3.0.2/src/zarr/core/common.py#L116-L130)) to valid Zarr V3 codec classes.
    - Parser changes are minimal.
    - Writer changes:
      - Kerchunk uses Zarr version format 2 so we convert `ArrayV3Metadata` to `ArrayV2Metadata` using the `convert_v3_to_v2_metadata` function. This means the `to_kerchunk_json` function is now a bit more complex because we're converting `ArrayV2Metadata` filters and compressor to serializable objects.
    - zarr-python 3.0 does not yet support the big endian data type. This means that FITS and NetCDF-3 are not currently supported ([zarr-python issue #2324](https://github.com/zarr-developers/zarr-python/issues/2324)).
    - zarr-python 3.0 does not yet support datetime and timedelta data types ([zarr-python issue #2616](https://github.com/zarr-developers/zarr-python/issues/2616)).
- The continuous integration workflows and developer environment now use [pixi](https://pixi.sh/latest/) ([#407](https://github.com/zarr-developers/VirtualiZarr/pull/407)).
- Added `loadable_variables` kwarg to `ManifestStore.to_virtual_dataset`.
  ([#543](https://github.com/zarr-developers/VirtualiZarr/pull/543)) By [Tom Nicholas](https://github.com/TomNicholas).
- Ensure that the `KerchunkJSONParser` can be used to parse in-memory kerchunk dictionaries using `obstore.store.MemoryStore`.
  ([#631](https://github.com/zarr-developers/VirtualiZarr/pull/631)) By [Tom Nicholas](https://github.com/TomNicholas).
- Move the `virtualizarr.translators.kerchunk` module to `virtualizarr.parsers.kerchunk.translator`, to better indicate that it is private. Also refactor the two kerchunk readers into one module.
  ([#633](https://github.com/zarr-developers/VirtualiZarr/pull/633)) By [Tom Nicholas](https://github.com/TomNicholas).

## v1.3.2 (3rd Mar 2025)

Small release which fixes a problem causing the docs to be out of date, fixes some issues in the tests with unclosed file handles, but also increases the performance of writing large numbers of virtual references to Icechunk!

### New Features

### Breaking changes

- Minimum supported version of Icechunk is now `v0.2.4` ([#462](https://github.com/zarr-developers/VirtualiZarr/pull/462))
  By [Tom Nicholas](https://github.com/TomNicholas).

### Deprecations

### Bug fixes

### Documentation

### Internal Changes

- Updates `store.set_virtual_ref` to `store.set_virtual_refs` in `write_manifest_virtual_refs` ([#443](https://github.com/zarr-developers/VirtualiZarr/pull/443)) By [Raphael Hagen](https://github.com/norlandrhagen).

## v1.3.1 (18th Feb 2025)

### New Features

- Examples use new Icechunk syntax

### Breaking changes

- Reading and writing Zarr chunk manifest formats are no longer supported.
  ([#359](https://github.com/zarr-developers/VirtualiZarr/issues/359)), ([#426](https://github.com/zarr-developers/VirtualiZarr/pull/426)). By [Raphael Hagen](https://github.com/norlandrhagen).

### Deprecations

### Bug fixes

### Documentation

### Internal Changes

## v1.3.0 (3rd Feb 2025)

This release stabilises our dependencies - you can now use released versions of VirtualiZarr, Kerchunk, and Icechunk all in the same environment!

It also fixes a number of bugs, adds minor features, changes the default reader for HDF/netCDF4 files, and includes refactors to reduce code redundancy with zarr-python v3. You can also choose which sets of dependencies you want at installation time.

### New Features

- Optional dependencies can now be installed in groups via pip. See the installation docs.
  ([#309](https://github.com/zarr-developers/VirtualiZarr/pull/309)) By [Tom Nicholas](https://github.com/TomNicholas).
- Added a `.nbytes` accessor method which displays the bytes needed to hold the virtual references in memory.
  ([#167](https://github.com/zarr-developers/VirtualiZarr/issues/167), [#227](https://github.com/zarr-developers/VirtualiZarr/pull/227)) By [Tom Nicholas](https://github.com/TomNicholas).
- Upgrade icechunk dependency to `>=0.1.0a12`. ([#406](https://github.com/zarr-developers/VirtualiZarr/pull/406)) By [Julia Signell](https://github.com/jsignell).
- Sync with Icechunk v0.1.0a8  ([#368](https://github.com/zarr-developers/VirtualiZarr/pull/368)) By [Matthew Iannucci](https://github.com/mpiannucci). This also adds support
  for the `to_icechunk` method to add timestamps as checksums when writing virtual references to an icechunk store. This
  is useful for ensuring that virtual references are not stale when reading from an icechunk store, which can happen if the
  underlying data has changed since the virtual references were written.
- Add `group=None` keyword-only parameter to the
  `VirtualiZarrDatasetAccessor.to_icechunk` method to allow writing to a nested group
  at a specified group path (rather than defaulting to the root group, when no group is
  specified).  ([#341](https://github.com/zarr-developers/VirtualiZarr/issues/341)) By [Chuck Daniels](https://github.com/chuckwondo).

### Breaking changes

- Passing `group=None` (the default) to `open_virtual_dataset` for a file with multiple groups no longer raises an error, instead it gives you the root group.
  This new behaviour is more consistent with `xarray.open_dataset`.
  ([#336](https://github.com/zarr-developers/VirtualiZarr/issues/336), [#338](https://github.com/zarr-developers/VirtualiZarr/pull/338)) By [Tom Nicholas](https://github.com/TomNicholas).
- Indexes are now created by default for any loadable one-dimensional coordinate variables.
  Also a warning is no longer thrown when `indexes=None` is passed to `open_virtual_dataset`, and the recommendations in the docs updated to match.
  This also means that `xarray.combine_by_coords` will now work when the necessary dimension coordinates are specified in `loadable_variables`.
  ([#18](https://github.com/zarr-developers/VirtualiZarr/issues/18), [#357](https://github.com/zarr-developers/VirtualiZarr/pull/357), [#358](https://github.com/zarr-developers/VirtualiZarr/pull/358)) By [Tom Nicholas](https://github.com/TomNicholas).
- The `append_dim` and `last_updated_at` parameters of the
  `VirtualiZarrDatasetAccessor.to_icechunk` method are now keyword-only parameters,
  rather than positional or keyword.  This change is breaking _only_ where arguments for
  these parameters are currently given positionally.  ([#341](https://github.com/zarr-developers/VirtualiZarr/issues/341)) By
  [Chuck Daniels](https://github.com/chuckwondo).
- The default backend for netCDF4 and HDF5 is now the custom `HDFVirtualBackend` replacing
  the previous default which was a wrapper around the kerchunk backend.
  ([#374](https://github.com/zarr-developers/VirtualiZarr/issues/374), [#395](https://github.com/zarr-developers/VirtualiZarr/pull/395)) By [Julia Signell](https://github.com/jsignell).
- Optional dependency on kerchunk is now the newly-released v0.2.8. This release of kerchunk is compatible with zarr-python v3.0.0,
  which means a released version of kerchunk can now be used with both VirtualiZarr and Icechunk.
  ([#392](https://github.com/zarr-developers/VirtualiZarr/issues/392), [#406](https://github.com/zarr-developers/VirtualiZarr/pull/406), [#412](https://github.com/zarr-developers/VirtualiZarr/pull/412)) By [Julia Signell](https://github.com/jsignell) and [Tom Nicholas](https://github.com/TomNicholas).

### Deprecations

### Bug fixes

- Fix bug preventing generating references for the root group of a file when a subgroup exists.
  ([#336](https://github.com/zarr-developers/VirtualiZarr/issues/336), [#338](https://github.com/zarr-developers/VirtualiZarr/pull/338)) By [Tom Nicholas](https://github.com/TomNicholas).
- Fix bug in HDF reader where dimension names of dimensions in a subgroup would be incorrect.
  ([#364](https://github.com/zarr-developers/VirtualiZarr/issues/364), [#366](https://github.com/zarr-developers/VirtualiZarr/pull/366)) By [Tom Nicholas](https://github.com/TomNicholas).
- Fix bug in dmrpp reader so _FillValue is included in variables' encodings.
  ([#369](https://github.com/zarr-developers/VirtualiZarr/pull/369)) By [Aimee Barciauskas](https://github.com/abarciauskas-bgse).
- Fix bug passing arguments to FITS reader, and test it on Hubble Space Telescope data.
  ([#363](https://github.com/zarr-developers/VirtualiZarr/pull/363)) By [Tom Nicholas](https://github.com/TomNicholas).

### Documentation

- Change intro text in readme and docs landing page to be clearer, less about the relationship to Kerchunk, and more about why you would want virtual datasets in the first place.
  ([#337](https://github.com/zarr-developers/VirtualiZarr/pull/337)) By [Tom Nicholas](https://github.com/TomNicholas).

### Internal Changes

- Add netCDF3 test. ([#397](https://github.com/zarr-developers/VirtualiZarr/pull/397)) By [Tom Nicholas](https://github.com/TomNicholas).

## v1.2.0 (5th Dec 2024)

This release brings a stricter internal model for manifest paths,
support for appending to existing icechunk stores,
an experimental non-kerchunk-based HDF5 reader,
handling of nested groups in DMR++ files,
as well as many other bugfixes and documentation improvements.

### New Features

- Add a `virtual_backend_kwargs` keyword argument to file readers and to `open_virtual_dataset`, to allow reader-specific options to be passed down.
  ([#315](https://github.com/zarr-developers/VirtualiZarr/pull/315)) By [Tom Nicholas](https://github.com/TomNicholas).
- Added append functionality to `to_icechunk` ([#272](https://github.com/zarr-developers/VirtualiZarr/pull/272)) By [Aimee Barciauskas](https://github.com/abarciauskas-bgse).

### Breaking changes

- Minimum required version of Xarray is now v2024.10.0.
  ([#284](https://github.com/zarr-developers/VirtualiZarr/pull/284)) By [Tom Nicholas](https://github.com/TomNicholas).
- Minimum required version of Icechunk is now v0.1.1.
  ([#419](https://github.com/zarr-developers/VirtualiZarr/pull/419)) By [Tom Nicholas](https://github.com/TomNicholas).
- Minimum required version of Kerchunk is now v0.2.8.
  ([#406](https://github.com/zarr-developers/VirtualiZarr/pull/406)) By [Julia Signell](https://github.com/jsignell).
- Opening kerchunk-formatted references from disk which contain relative paths now requires passing the `fs_root` keyword argument via `virtual_backend_kwargs`.
  ([#243](https://github.com/zarr-developers/VirtualiZarr/pull/243)) By [Tom Nicholas](https://github.com/TomNicholas).

### Deprecations

### Bug fixes

- Handle root and nested groups with `dmrpp` backend ([#265](https://github.com/zarr-developers/VirtualiZarr/pull/265))
  By [Ayush Nag](https://github.com/ayushnag).
- Fixed bug with writing of `dimension_names` into zarr metadata.
  ([#286](https://github.com/zarr-developers/VirtualiZarr/pull/286)) By [Tom Nicholas](https://github.com/TomNicholas).
- Fixed bug causing CF-compliant variables not to be identified as coordinates ([#191](https://github.com/zarr-developers/VirtualiZarr/pull/191))
  By [Ayush Nag](https://github.com/ayushnag).

### Documentation

- FAQ answers on Icechunk compatibility, converting from existing Kerchunk references to Icechunk, and how to add a new reader for a custom file format.
  ([#266](https://github.com/zarr-developers/VirtualiZarr/pull/266)) By [Tom Nicholas](https://github.com/TomNicholas).
- Clarify which readers actually currently work in FAQ, and temporarily remove tiff from the auto-detection.
  ([#291](https://github.com/zarr-developers/VirtualiZarr/issues/291), [#296](https://github.com/zarr-developers/VirtualiZarr/pull/296)) By [Tom Nicholas](https://github.com/TomNicholas).
- Minor improvements to the Contributing Guide.
  ([#298](https://github.com/zarr-developers/VirtualiZarr/pull/298)) By [Tom Nicholas](https://github.com/TomNicholas).
- More minor improvements to the Contributing Guide.
  ([#304](https://github.com/zarr-developers/VirtualiZarr/pull/304)) By [Doug Latornell](https://github.com/DougLatornell).
- Correct some links to the API.
  ([#325](https://github.com/zarr-developers/VirtualiZarr/pull/325)) By [Tom Nicholas](https://github.com/TomNicholas).
- Added links to recorded presentations on VirtualiZarr.
  ([#313](https://github.com/zarr-developers/VirtualiZarr/pull/313)) By [Tom Nicholas](https://github.com/TomNicholas).
- Added links to existing example notebooks.
  ([#329](https://github.com/zarr-developers/VirtualiZarr/issues/329), [#331](https://github.com/zarr-developers/VirtualiZarr/pull/331)) By [Tom Nicholas](https://github.com/TomNicholas).

### Internal Changes

- Added experimental new HDF file reader which doesn't use kerchunk, accessible by importing `virtualizarr.readers.hdf.HDFVirtualBackend`.
  ([#87](https://github.com/zarr-developers/VirtualiZarr/pull/87)) By [Sean Harkins](https://github.com/sharkinsspatial).
- Support downstream type checking by adding py.typed marker file.
  ([#306](https://github.com/zarr-developers/VirtualiZarr/pull/306)) By [Max Jones](https://github.com/maxrjones).
- File paths in chunk manifests are now always stored as absolute URIs.
  ([#243](https://github.com/zarr-developers/VirtualiZarr/pull/243)) By [Tom Nicholas](https://github.com/TomNicholas).

## v1.1.0 (22nd Oct 2024)

### New Features

- Can open `kerchunk` reference files with `open_virtual_dataset`.
  ([#251](https://github.com/zarr-developers/VirtualiZarr/pull/251), [#186](https://github.com/zarr-developers/VirtualiZarr/pull/186)) By [Raphael Hagen](https://github.com/norlandrhagen) & [Kristen Thyng](https://github.com/kthyng).
- Adds defaults for `open_virtual_dataset_from_v3_store` in ([#234](https://github.com/zarr-developers/VirtualiZarr/pull/234))
  By [Raphael Hagen](https://github.com/norlandrhagen).
- New `group` option on `open_virtual_dataset` enables extracting specific HDF Groups.
  ([#165](https://github.com/zarr-developers/VirtualiZarr/pull/165)) By [Scott Henderson](https://github.com/scottyhq).
- Adds `decode_times` to open_virtual_dataset ([#232](https://github.com/zarr-developers/VirtualiZarr/pull/232))
  By [Raphael Hagen](https://github.com/norlandrhagen).
- Add parser for the OPeNDAP DMR++ XML format and integration with open_virtual_dataset ([#113](https://github.com/zarr-developers/VirtualiZarr/pull/113))
  By [Ayush Nag](https://github.com/ayushnag).
- Load scalar variables by default. ([#205](https://github.com/zarr-developers/VirtualiZarr/pull/205))
  By [Gustavo Hidalgo](https://github.com/ghidalgo3).
- Support empty files ([#260](https://github.com/zarr-developers/VirtualiZarr/pull/260))
  By [Justus Magin](https://github.com/keewis).
- Can write virtual datasets to Icechunk stores using `virtualize.to_icechunk` ([#256](https://github.com/zarr-developers/VirtualiZarr/pull/256))
  By [Matt Iannucci](https://github.com/mpiannucci).

### Breaking changes

- Serialize valid ZarrV3 metadata and require full compressor numcodec config (for [#193](https://github.com/zarr-developers/VirtualiZarr/pull/193))
  By [Gustavo Hidalgo](https://github.com/ghidalgo3).
- VirtualiZarr's `ZArray`, `ChunkEntry`, and `Codec` no longer subclass
  `pydantic.BaseModel` ([#210](https://github.com/zarr-developers/VirtualiZarr/pull/210))
- `ZArray`'s `__init__` signature has changed to match `zarr.Array`'s ([#210](https://github.com/zarr-developers/VirtualiZarr/pull/210))

### Deprecations

- Depreciates cftime_variables in open_virtual_dataset in favor of decode_times. ([#232](https://github.com/zarr-developers/VirtualiZarr/pull/232))
  By [Raphael Hagen](https://github.com/norlandrhagen).

### Bug fixes

- Exclude empty chunks during `ChunkDict` construction. ([#198](https://github.com/zarr-developers/VirtualiZarr/pull/198))
  By [Gustavo Hidalgo](https://github.com/ghidalgo3).
- Fixed regression in `fill_value` handling for datetime dtypes making virtual
  Zarr stores unreadable ([#206](https://github.com/zarr-developers/VirtualiZarr/pull/206))
  By [Timothy Hodson](https://github.com/thodson-usgs)

### Documentation

- Adds virtualizarr + coiled serverless example notebook ([#223](https://github.com/zarr-developers/VirtualiZarr/pull/223))
  By [Raphael Hagen](https://github.com/norlandrhagen).

### Internal Changes

- Refactored internal structure significantly to split up everything to do with reading references from that to do with writing references.
  ([#229](https://github.com/zarr-developers/VirtualiZarr/issues/229)) ([#231](https://github.com/zarr-developers/VirtualiZarr/pull/231)) By [Tom Nicholas](https://github.com/TomNicholas).
- Refactored readers to consider every filetype as a separate reader, all standardized to present the same `open_virtual_dataset` interface internally.
  ([#261](https://github.com/zarr-developers/VirtualiZarr/pull/261)) By [Tom Nicholas](https://github.com/TomNicholas).

## v1.0.0 (9th July 2024)

This release marks VirtualiZarr as mostly feature-complete, in the sense of achieving feature parity with kerchunk's logic for combining datasets, providing an easier way to manipulate kerchunk references in memory and generate kerchunk reference files on disk.

Future VirtualiZarr development will focus on generalizing and upstreaming useful concepts into the Zarr specification, the Zarr-Python library, Xarray, and possibly some new packages. See the roadmap in the documentation for details.

### New Features

- Now successfully opens both tiff and FITS files. ([#160](https://github.com/zarr-developers/VirtualiZarr/issues/160), [#162](https://github.com/zarr-developers/VirtualiZarr/pull/162))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Added a `.rename_paths` convenience method to rename paths in a manifest according to a function.
  ([#152](https://github.com/zarr-developers/VirtualiZarr/pull/152)) By [Tom Nicholas](https://github.com/TomNicholas).
- New `cftime_variables` option on `open_virtual_dataset` enables encoding/decoding time.
  ([#122](https://github.com/zarr-developers/VirtualiZarr/pull/122)) By [Julia Signell](https://github.com/jsignell).

### Breaking changes

- Requires numpy 2.0 (for [#107](https://github.com/zarr-developers/VirtualiZarr/pull/107)).
  By [Tom Nicholas](https://github.com/TomNicholas).

### Deprecations

### Bug fixes

- Ensure that `_ARRAY_DIMENSIONS` are dropped from variable `.attrs`. ([#150](https://github.com/zarr-developers/VirtualiZarr/issues/150), [#152](https://github.com/zarr-developers/VirtualiZarr/pull/152))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Ensure that `.attrs` on coordinate variables are preserved during round-tripping. ([#155](https://github.com/zarr-developers/VirtualiZarr/issues/155), [#154](https://github.com/zarr-developers/VirtualiZarr/pull/154))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Ensure that non-dimension coordinate variables described via the CF conventions are preserved during round-tripping. ([#105](https://github.com/zarr-developers/VirtualiZarr/issues/105), [#156](https://github.com/zarr-developers/VirtualiZarr/pull/156))
  By [Tom Nicholas](https://github.com/TomNicholas).

### Documentation

- Added example of using cftime_variables to usage docs. ([#169](https://github.com/zarr-developers/VirtualiZarr/issues/169), [#174](https://github.com/zarr-developers/VirtualiZarr/pull/174))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Updated the development roadmap in preparation for v1.0. ([#164](https://github.com/zarr-developers/VirtualiZarr/pull/164))
  By [Tom Nicholas](https://github.com/TomNicholas).
- Warn if user passes `indexes=None` to `open_virtual_dataset` to indicate that this is not yet fully supported.
  ([#170](https://github.com/zarr-developers/VirtualiZarr/pull/170)) By [Tom Nicholas](https://github.com/TomNicholas).
- Clarify that virtual datasets cannot be treated like normal xarray datasets. ([#173](https://github.com/zarr-developers/VirtualiZarr/issues/173))
  By [Tom Nicholas](https://github.com/TomNicholas).

### Internal Changes

- Refactor `ChunkManifest` class to store chunk references internally using numpy arrays.
  ([#107](https://github.com/zarr-developers/VirtualiZarr/pull/107)) By [Tom Nicholas](https://github.com/TomNicholas).
- Mark tests which require network access so that they are only run when `--run-network-tests` is passed a command-line argument to pytest.
  ([#144](https://github.com/zarr-developers/VirtualiZarr/pull/144)) By [Tom Nicholas](https://github.com/TomNicholas).
- Determine file format from magic bytes rather than name suffix
  ([#143](https://github.com/zarr-developers/VirtualiZarr/pull/143)) By [Scott Henderson](https://github.com/scottyhq).

## v0.1 (17th June 2024)

v0.1 is the first release of VirtualiZarr!! It contains functionality for using kerchunk to find byte ranges in netCDF files,
constructing an xarray.Dataset containing ManifestArray objects, then writing out such a dataset to kerchunk references as either json or parquet.

### New Features

### Breaking changes

### Deprecations

### Bug fixes

### Documentation

### Internal Changes
