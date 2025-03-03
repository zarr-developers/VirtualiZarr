Release notes
=============

.. _v1.3.2:

v1.3.2 (3rd Mar 2025)
---------------------

Small release which fixes a problem causing the docs to be out of date, fixes some issues in the tests with unclosed file handles, but also increases the performance of writing large numbers of virtual references to Icechunk!

New Features
~~~~~~~~~~~~

Breaking changes
~~~~~~~~~~~~~~~~

- Minimum supported version of Icechunk is now `v0.2.4` (:pull:`462`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

- Updates `store.set_virtual_ref` to `store.set_virtual_refs` in `write_manifest_virtual_refs` (:pull:`443`) By `Raphael Hagen <https://github.com/norlandrhagen>`_.

.. _v1.3.1:

v1.3.1 (18th Feb 2025)
----------------------

New Features
~~~~~~~~~~~~

- Examples use new Icechunk syntax

Breaking changes
~~~~~~~~~~~~~~~~

- Reading and writing Zarr chunk manifest formats are no longer supported.
  (:issue:`359`), (:pull:`426`). By `Raphael Hagen <https://github.com/norlandrhagen>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

Documentation
~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~

.. _v1.3.0:

v1.3.0 (3rd Feb 2025)
---------------------

This release stabilises our dependencies - you can now use released versions of VirtualiZarr, Kerchunk, and Icechunk all in the same environment!

It also fixes a number of bugs, adds minor features, changes the default reader for HDF/netCDF4 files, and includes refactors to reduce code redundancy with zarr-python v3. You can also choose which sets of dependencies you want at installation time.

New Features
~~~~~~~~~~~~

- Optional dependencies can now be installed in groups via pip. See the installation docs.
  (:pull:`309`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a ``.nbytes`` accessor method which displays the bytes needed to hold the virtual references in memory.
  (:issue:`167`, :pull:`227`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Upgrade icechunk dependency to ``>=0.1.0a12``. (:pull:`406`) By `Julia Signell <https://github.com/jsignell>`_.
- Sync with Icechunk v0.1.0a8  (:pull:`368`) By `Matthew Iannucci <https://github.com/mpiannucci>`. This also adds support
  for the `to_icechunk` method to add timestamps as checksums when writing virtual references to an icechunk store. This
  is useful for ensuring that virtual references are not stale when reading from an icechunk store, which can happen if the
  underlying data has changed since the virtual references were written.
- Add ``group=None`` keyword-only parameter to the
  ``VirtualiZarrDatasetAccessor.to_icechunk`` method to allow writing to a nested group
  at a specified group path (rather than defaulting to the root group, when no group is
  specified).  (:issue:`341`) By `Chuck Daniels <https://github.com/chuckwondo>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Passing ``group=None`` (the default) to ``open_virtual_dataset`` for a file with multiple groups no longer raises an error, instead it gives you the root group.
  This new behaviour is more consistent with ``xarray.open_dataset``.
  (:issue:`336`, :pull:`338`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Indexes are now created by default for any loadable one-dimensional coordinate variables.
  Also a warning is no longer thrown when ``indexes=None`` is passed to ``open_virtual_dataset``, and the recommendations in the docs updated to match.
  This also means that ``xarray.combine_by_coords`` will now work when the necessary dimension coordinates are specified in ``loadable_variables``.
  (:issue:`18`, :pull:`357`, :pull:`358`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- The ``append_dim`` and ``last_updated_at`` parameters of the
  ``VirtualiZarrDatasetAccessor.to_icechunk`` method are now keyword-only parameters,
  rather than positional or keyword.  This change is breaking _only_ where arguments for
  these parameters are currently given positionally.  (:issue:`341`) By
  `Chuck Daniels <https://github.com/chuckwondo>`_.
- The default backend for netCDF4 and HDF5 is now the custom ``HDFVirtualBackend`` replacing
  the previous default which was a wrapper around the kerchunk backend.
  (:issue:`374`, :pull:`395`) By `Julia Signell <https://github.com/jsignell>`_.
- Optional dependency on kerchunk is now the newly-released v0.2.8. This release of kerchunk is compatible with zarr-python v3.0.0,
  which means a released version of kerchunk can now be used with both VirtualiZarr and Icechunk.
  (:issue:`392`, :pull:`406`, :pull:`412``) By `Julia Signell <https://github.com/jsignell>`_ and `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Fix bug preventing generating references for the root group of a file when a subgroup exists.
  (:issue:`336`, :pull:`338`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix bug in HDF reader where dimension names of dimensions in a subgroup would be incorrect.
  (:issue:`364`, :pull:`366`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fix bug in dmrpp reader so _FillValue is included in variables' encodings.
  (:pull:`369`) By `Aimee Barciauskas <https://github.com/abarciauskas-bgse>`_.
- Fix bug passing arguments to FITS reader, and test it on Hubble Space Telescope data.
  (:pull:`363`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Change intro text in readme and docs landing page to be clearer, less about the relationship to Kerchunk, and more about why you would want virtual datasets in the first place.
  (:pull:`337`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Add netCDF3 test. (:pull:`397`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _v1.2.0:

v1.2.0 (5th Dec 2024)
---------------------

This release brings a stricter internal model for manifest paths,
support for appending to existing icechunk stores,
an experimental non-kerchunk-based HDF5 reader,
handling of nested groups in DMR++ files,
as well as many other bugfixes and documentation improvements.

New Features
~~~~~~~~~~~~

- Add a ``virtual_backend_kwargs`` keyword argument to file readers and to ``open_virtual_dataset``, to allow reader-specific options to be passed down.
  (:pull:`315`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added append functionality to `to_icechunk` (:pull:`272`) By `Aimee Barciauskas <https://github.com/abarciauskas-bgse>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Minimum required version of Xarray is now v2024.10.0.
  (:pull:`284`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Minimum required version of Icechunk is now v0.1.1.
  (:pull:`419`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Minimum required version of Kerchunk is now v0.2.8.
  (:pull:`406`) By `Julia Signell <https://github.com/jsignell>`_.
- Opening kerchunk-formatted references from disk which contain relative paths now requires passing the ``fs_root`` keyword argument via ``virtual_backend_kwargs``.
  (:pull:`243`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~

Bug fixes
~~~~~~~~~

- Handle root and nested groups with ``dmrpp`` backend (:pull:`265`)
  By `Ayush Nag <https://github.com/ayushnag>`_.
- Fixed bug with writing of `dimension_names` into zarr metadata.
  (:pull:`286`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Fixed bug causing CF-compliant variables not to be identified as coordinates (:pull:`191`)
  By `Ayush Nag <https://github.com/ayushnag>`_.

Documentation
~~~~~~~~~~~~~

- FAQ answers on Icechunk compatibility, converting from existing Kerchunk references to Icechunk, and how to add a new reader for a custom file format.
  (:pull:`266`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Clarify which readers actually currently work in FAQ, and temporarily remove tiff from the auto-detection.
  (:issue:`291`, :pull:`296`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Minor improvements to the Contributing Guide.
  (:pull:`298`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- More minor improvements to the Contributing Guide.
  (:pull:`304`) By `Doug Latornell <https://github.com/DougLatornell>`_.
- Correct some links to the API.
  (:pull:`325`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added links to recorded presentations on VirtualiZarr.
  (:pull:`313`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added links to existing example notebooks.
  (:issue:`329`, :pull:`331`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Added experimental new HDF file reader which doesn't use kerchunk, accessible by importing ``virtualizarr.readers.hdf.HDFVirtualBackend``.
  (:pull:`87`) By `Sean Harkins <https://github.com/sharkinsspatial>`_.
- Support downstream type checking by adding py.typed marker file.
  (:pull:`306`) By `Max Jones <https://github.com/maxrjones>`_.
- File paths in chunk manifests are now always stored as abolute URIs.
  (:pull:`243`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _v1.1.0:

v1.1.0 (22nd Oct 2024)
----------------------

New Features
~~~~~~~~~~~~

- Can open `kerchunk` reference files with ``open_virtual_dataset``.
  (:pull:`251`, :pull:`186`) By `Raphael Hagen <https://github.com/norlandrhagen>`_ & `Kristen Thyng <https://github.com/kthyng>`_.
- Adds defaults for `open_virtual_dataset_from_v3_store` in (:pull:`234`)
  By `Raphael Hagen <https://github.com/norlandrhagen>`_.
- New ``group`` option on ``open_virtual_dataset`` enables extracting specific HDF Groups.
  (:pull:`165`) By `Scott Henderson <https://github.com/scottyhq>`_.
- Adds `decode_times` to open_virtual_dataset (:pull:`232`)
  By `Raphael Hagen <https://github.com/norlandrhagen>`_.
- Add parser for the OPeNDAP DMR++ XML format and integration with open_virtual_dataset (:pull:`113`)
  By `Ayush Nag <https://github.com/ayushnag>`_.
- Load scalar variables by default. (:pull:`205`)
  By `Gustavo Hidalgo <https://github.com/ghidalgo3>`_.
- Support empty files (:pull:`260`)
  By `Justus Magin <https://github.com/keewis>`_.
- Can write virtual datasets to Icechunk stores using `vitualize.to_icechunk` (:pull:`256`)
  By `Matt Iannucci <https://github.com/mpiannucci>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Serialize valid ZarrV3 metadata and require full compressor numcodec config (for :pull:`193`)
  By `Gustavo Hidalgo <https://github.com/ghidalgo3>`_.
- VirtualiZarr's `ZArray`, `ChunkEntry`, and `Codec` no longer subclass
  `pydantic.BaseModel` (:pull:`210`)
- `ZArray`'s `__init__` signature has changed to match `zarr.Array`'s (:pull:`210`)

Deprecations
~~~~~~~~~~~~

- Depreciates cftime_variables in open_virtual_dataset in favor of decode_times. (:pull:`232`)
  By `Raphael Hagen <https://github.com/norlandrhagen>`_.

Bug fixes
~~~~~~~~~

- Exclude empty chunks during `ChunkDict` construction. (:pull:`198`)
  By `Gustavo Hidalgo <https://github.com/ghidalgo3>`_.
- Fixed regression in `fill_value` handling for datetime dtypes making virtual
  Zarr stores unreadable (:pull:`206`)
  By `Timothy Hodson <https://github.com/thodson-usgs>`_

Documentation
~~~~~~~~~~~~~

- Adds virtualizarr + coiled serverless example notebook (:pull:`223`)
  By `Raphael Hagen <https://github.com/norlandrhagen>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Refactored internal structure significantly to split up everything to do with reading references from that to do with writing references.
  (:issue:`229`) (:pull:`231`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Refactored readers to consider every filetype as a separate reader, all standardized to present the same `open_virtual_dataset` interface internally.
  (:pull:`261`) By `Tom Nicholas <https://github.com/TomNicholas>`_.

.. _v1.0.0:

v1.0.0 (9th July 2024)
----------------------

This release marks VirtualiZarr as mostly feature-complete, in the sense of achieving feature parity with kerchunk's logic for combining datasets, providing an easier way to manipulate kerchunk references in memory and generate kerchunk reference files on disk.

Future VirtualiZarr development will focus on generalizing and upstreaming useful concepts into the Zarr specification, the Zarr-Python library, Xarray, and possibly some new packages. See the roadmap in the documentation for details.

New Features
~~~~~~~~~~~~

- Now successfully opens both tiff and FITS files. (:issue:`160`, :pull:`162`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Added a `.rename_paths` convenience method to rename paths in a manifest according to a function.
  (:pull:`152`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- New ``cftime_variables`` option on ``open_virtual_dataset`` enables encoding/decoding time.
  (:pull:`122`) By `Julia Signell <https://github.com/jsignell>`_.

Breaking changes
~~~~~~~~~~~~~~~~

- Requires numpy 2.0 (for :pull:`107`).
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Deprecations
~~~~~~~~~~~~


Bug fixes
~~~~~~~~~

- Ensure that `_ARRAY_DIMENSIONS` are dropped from variable `.attrs`. (:issue:`150`, :pull:`152`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Ensure that `.attrs` on coordinate variables are preserved during round-tripping. (:issue:`155`, :pull:`154`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Ensure that non-dimension coordinate variables described via the CF conventions are preserved during round-tripping. (:issue:`105`, :pull:`156`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Documentation
~~~~~~~~~~~~~

- Added example of using cftime_variables to usage docs. (:issue:`169`, :pull:`174`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Updated the development roadmap in preparation for v1.0. (:pull:`164`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Warn if user passes `indexes=None` to `open_virtual_dataset` to indicate that this is not yet fully supported.
  (:pull:`170`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Clarify that virtual datasets cannot be treated like normal xarray datasets. (:issue:`173`)
  By `Tom Nicholas <https://github.com/TomNicholas>`_.

Internal Changes
~~~~~~~~~~~~~~~~

- Refactor `ChunkManifest` class to store chunk references internally using numpy arrays.
  (:pull:`107`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Mark tests which require network access so that they are only run when `--run-network-tests` is passed a command-line argument to pytest.
  (:pull:`144`) By `Tom Nicholas <https://github.com/TomNicholas>`_.
- Determine file format from magic bytes rather than name suffix
  (:pull:`143`) By `Scott Henderson <https://github.com/scottyhq>`_.

.. _v0.1:

v0.1 (17th June 2024)
---------------------

v0.1 is the first release of VirtualiZarr!! It contains functionality for using kerchunk to find byte ranges in netCDF files,
constructing an xarray.Dataset containing ManifestArray objects, then writing out such a dataset to kerchunk references as either json or parquet.

New Features
~~~~~~~~~~~~


Breaking changes
~~~~~~~~~~~~~~~~


Deprecations
~~~~~~~~~~~~


Bug fixes
~~~~~~~~~


Documentation
~~~~~~~~~~~~~


Internal Changes
~~~~~~~~~~~~~~~~
