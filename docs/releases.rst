Release notes
=============

.. _v0.2:

v0.2 (unreleased)
-----------------

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
