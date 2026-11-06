# Supported file formats

VirtualiZarr reads each file format through a [parser](explanation/custom_parsers.md).
Some parsers ship with VirtualiZarr, and others live in separate packages.
Many formats are another format underneath. For example, NetCDF4, AnnData (`.h5ad`) and MATLAB v7.3 (`.mat`) files are all HDF5 files.
The table below lists each format under the name you probably know it by, so you can search this page for your file extension.

!!! tip "Missing your format? Help us grow this list!"

    If you know of a parser that isn't listed here, or a format that turns out to be one of these formats underneath,
    please [edit this page](https://github.com/zarr-developers/VirtualiZarr/edit/main/docs/supported_formats.md) and open a pull request.
    You can also suggest formats on [issue #218](https://github.com/zarr-developers/VirtualiZarr/issues/218).
    To write a parser for a new format, see [Custom Parsers](explanation/custom_parsers.md): it can live in your own package.

**Columns:**

- **Parser**: the class to pass as `open_virtual_dataset(..., parser=...)`.
- **Package**: where the parser comes from, with the `pip install` extra if there is one.
- **VZ team?**: ✅ if a member of the [VirtualiZarr core team](https://github.com/orgs/zarr-developers/teams/virtualizarr-core-devs) maintains the parser.
- **Codecs needed to read**: what anyone *reading* the virtual store must have installed to decode chunks, beyond zarr-python. "None" means only codecs from the core Zarr V3 spec. `numcodecs` codecs are available in zarr-python, but readers in other languages may not support them. A named package means its codecs must be installed and registered wherever the data is read, and readers in other languages may not support them.
- **Domains**: the fields where the format is most common.

## Formats with a parser

| Format | Extensions | Parser | Package | VZ team? | Codecs needed to read | Domains | Notes |
| --- | --- | --- | --- | :---: | --- | --- | --- |
| AnnData | `.h5ad` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">bioinformatics</span> | HDF5 underneath. Sparse matrices are stored as several 1D arrays (`data`, `indices`, `indptr`), which are virtualized separately. |
| BigTIFF | `.tif`, `.tiff`, `.btf` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">general</span> | TIFF with 64-bit offsets. |
| Cloud-Optimized GeoTIFF (COG) | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">geospatial</span> <span class="fmt-tag">remote sensing</span> | TIFF. Overviews can be exposed as a Zarr multiscale group. |
| DMR++ | `.dmrpp` | [`DMRPPParser`][virtualizarr.parsers.DMRPPParser] | `virtualizarr` | ✅ | `numcodecs` | <span class="fmt-tag">earth observation</span> | OPeNDAP sidecar file that indexes an HDF5/NetCDF4 file. Common in NASA Earthdata archives. Using the sidecar avoids reading the HDF5 file's metadata. |
| ENVI | `.hdr` with `.img`, `.dat`, `.bsq`, `.bil`, `.bip`, or no extension | [`ENVIParser`][virtualizarr.parsers.ENVIParser] | `virtualizarr` | ✅ | None | <span class="fmt-tag">remote sensing</span> <span class="fmt-tag">hyperspectral</span> | Prototype. Only the "ENVI Standard" file type is supported, in BSQ, BIL or BIP interleave. A whole-file gzipped data file becomes a single chunk. |
| FITS | `.fits`, `.fit`, `.fts` | [`FITSParser`][virtualizarr.parsers.FITSParser] | `virtualizarr[fits]` | ✅ | `numcodecs` for scaled (`BSCALE`/`BZERO`) images. `virtualizarr` for ASCII tables. | <span class="fmt-tag">astronomy</span> | Image/cube HDUs and ASCII tables are supported. Binary tables (`BINTABLE`) are not yet supported. |
| GeoTIFF | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">geospatial</span> <span class="fmt-tag">remote sensing</span> | TIFF. |
| GRIB1 / GRIB2 | `.grib`, `.grb`, `.grib2`, `.grb2` | [`GribberishParser`](api/parsers/grib.md) | [gribberish](https://github.com/mpiannucci/gribberish) (`virtualizarr[grib]`) | ✅ | `gribberish` | <span class="fmt-tag">weather & climate</span> | Covers NWP model output such as HRRR and NBM. Each GRIB message becomes one chunk, decoded by a gribberish codec. GRIB1 complex and spherical-harmonic packing are not yet supported. Files whose messages don't form one hypercube need `open_virtual_datatree`. |
| HDF-EOS2 | `.hdf` | [`HDF4Parser`][virtualizarr.parsers.HDF4Parser] | `virtualizarr[hdf4]` | ✅ | `numcodecs` | <span class="fmt-tag">earth observation</span> | HDF4 underneath, e.g. older MODIS products. |
| HDF-EOS5 | `.he5`, `.h5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">earth observation</span> | HDF5 underneath, e.g. Aura and OMI products. |
| HDF4 | `.hdf`, `.h4`, `.hdf4` | [`HDF4Parser`][virtualizarr.parsers.HDF4Parser] | `virtualizarr[hdf4]` | ✅ | `numcodecs` | <span class="fmt-tag">earth observation</span> | Uses kerchunk internally. |
| HDF5 | `.h5`, `.hdf5`, `.he5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">general</span> | HDF5 links and object references are opt-in: see [HDF5 links and references](how_to/hdf5_links_references.md). |
| HEC-RAS | `.hdf` (`.p##.hdf`, `.g##.hdf`) | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">hydrology</span> | HDF5 underneath. The HDF5 contents can be virtualized, but the model-specific mesh structure isn't interpreted. |
| Icechunk | (a repository, not a file) | [`IcechunkParser`][virtualizarr.parsers.IcechunkParser] | `virtualizarr[icechunk]` | ✅ | Same as the source data | <span class="fmt-tag">general</span> | Re-virtualizes an existing Icechunk repository without copying data. Requires `icechunk>=2.0.5`. |
| Imaris | `.ims` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">microscopy</span> | HDF5 underneath. |
| Kerchunk references (JSON) | `.json` | [`KerchunkJSONParser`][virtualizarr.parsers.KerchunkJSONParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag">general</span> | Reads references you already made with kerchunk, so you can convert them to Icechunk. |
| Kerchunk references (Parquet) | `.parquet` (directory) | [`KerchunkParquetParser`][virtualizarr.parsers.KerchunkParquetParser] | `virtualizarr[kerchunk_parquet]` | ✅ | Same as the source data | <span class="fmt-tag">general</span> | As above. |
| MATLAB v7.3 | `.mat` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">general</span> | Only v7.3 `.mat` files are HDF5. Older `.mat` versions are a different format and aren't supported ([#1122](https://github.com/zarr-developers/VirtualiZarr/issues/1122)). |
| NetCDF3 (classic, 64-bit offset, CDF-5) | `.nc`, `.cdf` | [`NetCDF3Parser`][virtualizarr.parsers.NetCDF3Parser] | `virtualizarr` | ✅ | None | <span class="fmt-tag">weather & climate</span> <span class="fmt-tag">ocean</span> | All variables are uncompressed. Record (unlimited-dimension) variables are interleaved on disk, so each record becomes its own chunk. |
| NetCDF4 | `.nc`, `.nc4` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">weather & climate</span> <span class="fmt-tag">ocean</span> | HDF5 underneath. Check which one you have with `ncdump -k`: `netCDF-4` means HDF5, `classic` means NetCDF3. |
| NeXus | `.nxs`, `.h5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag">neutron & x-ray science</span> | HDF5 underneath. |
| OME-TIFF | `.ome.tif`, `.ome.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">microscopy</span> | TIFF. |
| OME-Zarr | `.ome.zarr`, `.zarr` | [`ZarrParser`][virtualizarr.parsers.ZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag">microscopy</span> <span class="fmt-tag">bioimaging</span> | Zarr. |
| TIFF | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">general</span> | Internally tiled TIFFs work well, even if they aren't COG-compliant. |
| Whole-slide images (e.g. Aperio) | `.svs`, `.tif` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag">pathology</span> <span class="fmt-tag">microscopy</span> | TIFF-based variants only. |
| Zarr V2 / V3 | `.zarr` (directory) | [`ZarrParser`][virtualizarr.parsers.ZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag">general</span> | Creates a "Zarr of Zarrs", useful for combining many existing stores into one. |
| Zipped Zarr | `.zarr.zip`, `.zip` | [`ZippedZarrParser`][virtualizarr.parsers.ZippedZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag">general</span> | Reads the archive's central directory, so the zip is never extracted. Entries must be stored uncompressed (`ZIP_STORED`). |

## Formats without a parser yet

These formats could probably be virtualized, but nobody has written a parser yet. If you need one, contributions are welcome. Comment on the linked issue, or write your own parser.

| Format | Extensions | Domains | Notes |
| --- | --- | --- | --- |
| DICOM | `.dcm` | <span class="fmt-tag">medical imaging</span> | Widespread, but complicated ([#218](https://github.com/zarr-developers/VirtualiZarr/issues/218)). |
| Hugging Face safetensors | `.safetensors` | <span class="fmt-tag">machine learning</span> | Uncompressed, with all metadata at the start of the file ([#367](https://github.com/zarr-developers/VirtualiZarr/issues/367)). |
| MATLAB (pre-v7.3) | `.mat` | <span class="fmt-tag">general</span> | See [#1122](https://github.com/zarr-developers/VirtualiZarr/issues/1122). For v7.3 files, use `HDFParser` (see above). |
| NASA CDF | `.cdf` | <span class="fmt-tag">space physics</span> | Not to be confused with netCDF. |
| NIfTI | `.nii` | <span class="fmt-tag">medical imaging</span> | Uncompressed `.nii` should be simple. `.nii.gz` is gzipped as a whole file (see below). |
| NumPy | `.npy`, `.npz` | <span class="fmt-tag">general</span> | `.npy` is a header followed by raw bytes. `.npz` is a zip of `.npy` files. |
| Open-Meteo OM files | `.om` | <span class="fmt-tag">weather & climate</span> | See [open-meteo/om-file-format#24](https://github.com/open-meteo/om-file-format/issues/24). |
| PDS4 | `.xml` with `.img`, `.dat` | <span class="fmt-tag">planetary science</span> | An XML label describing raw binary data. |
| RPN / FST | `.fst` | <span class="fmt-tag">weather & climate</span> | Environment and Climate Change Canada's model output format. |
| SAFE | `.SAFE` (directory) | <span class="fmt-tag">remote sensing</span> | Copernicus Sentinel product packaging. Its contents are often JP2 (see below) or GeoTIFF. |

## Formats that probably can't be virtualized

A format can only be virtualized if each chunk of each array is a contiguous byte range that a cheap codec can decode.
See [Can my file format be virtualized?](explanation/faq.md#can-my-file-format-be-virtualized) for why.

| Format | Domains | Why not |
| --- | --- | --- |
| Any whole-file-compressed file (e.g. `.nc.gz`, `.h5.bz2`, `.nii.gz`) | <span class="fmt-tag">general</span> | Compressing the whole file mixes up the bytes of every chunk. Decompress first, or (for small files) treat the whole file as one chunk. |
| CSV | <span class="fmt-tag">general</span> | A text format: every read needs expensive parsing, and rows don't map onto fixed-size chunks ([#200](https://github.com/zarr-developers/VirtualiZarr/issues/200)). |
| Furuno, Sigmet/IRIS, NEXRAD Level II | <span class="fmt-tag">weather radar</span> | Variables are interleaved within records. |
| GDAL Virtual Raster (`.vrt`) | <span class="fmt-tag">geospatial</span> | A VRT can resample, reproject and mosaic its source rasters on the fly, which plain byte-range references can't express ([#166](https://github.com/zarr-developers/VirtualiZarr/issues/166)). Virtualize the underlying GeoTIFFs instead. |
| JPEG 2000 | <span class="fmt-tag">geospatial</span> <span class="fmt-tag">remote sensing</span> | Probably not: see [GDAL's notes on JPEG 2000 access](https://gdal.org/user/virtual_file_systems.html#jpeg2000) and [#218](https://github.com/zarr-developers/VirtualiZarr/issues/218). |
| Parquet | <span class="fmt-tag">tabular</span> | Encodings such as dictionary pages can't be decoded from one contiguous byte range ([#441](https://github.com/zarr-developers/VirtualiZarr/issues/441)). |
| VCF | <span class="fmt-tag">genomics</span> | A text format with interleaved columns. Convert it to Zarr with [bio2zarr](https://github.com/sgkit-dev/bio2zarr) instead. |
