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
- **VZ team?**: ✅ if a member of the [VirtualiZarr core team](https://github.com/orgs/zarr-developers/teams/virtualizarr-core-devs) maintains the parser. ❌ means we can't vouch for its correctness (see the warning below).
- **Codecs needed to read**: what anyone *reading* the virtual store must have installed to decode chunks, beyond zarr-python. "None" means only codecs from the core Zarr V3 spec. `numcodecs` codecs are available in zarr-python, but readers in other languages may not support them. A named package means its codecs must be installed and registered wherever the data is read, and readers in other languages may not support them.
- **Domains**: the fields where the format is most common.

## ✅ Formats with a parser

!!! note "Parsing one file doesn't mean your whole dataset can be virtualized"

    A parser for your format means each file can be virtualized on its own.
    Combining many files into one virtual dataset also needs the files to be consistent with each other, for example in chunk shapes, codecs and CF encoding attributes.
    See [Can my specific data be virtualized?](explanation/faq.md#can-my-specific-data-be-virtualized) for the full list of requirements.

!!! warning "Check the output of parsers marked ❌"

    Parsers marked ❌ in the **VZ team?** column are maintained outside the VirtualiZarr team.
    We haven't reviewed them, so we can't vouch for their correctness.
    A parser bug can produce wrong chunk references that still open without errors and then return incorrect data.
    Before relying on one of these parsers, check its output carefully, for example by comparing values read through the virtual store with values read from the original file by a trusted library.

| Format | Extensions | Parser | Package | VZ team? | Codecs needed to read | Domains | Notes |
| --- | --- | --- | --- | :---: | --- | --- | --- |
| AnnData | `.h5ad` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-life">🧮 bioinformatics</span> | HDF5 underneath. Sparse matrices are stored as several 1D arrays (`data`, `indices`, `indptr`), which are virtualized separately. |
| BigTIFF | `.tif`, `.tiff`, `.btf` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-data">📦 general</span> | TIFF with 64-bit offsets. |
| Cloud-Optimized GeoTIFF (COG) | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-earth">🗺️ geospatial</span> <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> | TIFF. Overviews can be exposed as a Zarr multiscale group. Also readable with [`OversightMLParser`](https://awslabs.github.io/osml-imagery-io/api/virtualizarr-parsers.html) from osml-imagery-io. |
| DMR++ | `.dmrpp` | [`DMRPPParser`][virtualizarr.parsers.DMRPPParser] | `virtualizarr` | ✅ | `numcodecs` | <span class="fmt-tag fmt-earth">🌍 earth observation</span> | OPeNDAP sidecar file that indexes an HDF5/NetCDF4 file. Common in NASA Earthdata archives. Using the sidecar avoids reading the HDF5 file's metadata. |
| ENVI | `.hdr` with `.img`, `.dat`, `.bsq`, `.bil`, `.bip`, or no extension | [`ENVIParser`][virtualizarr.parsers.ENVIParser] | `virtualizarr` | ✅ | None | <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> <span class="fmt-tag fmt-earth">🌈 hyperspectral</span> | Prototype. Only the "ENVI Standard" file type is supported, in BSQ, BIL or BIP interleave. A whole-file gzipped data file becomes a single chunk. |
| FITS | `.fits`, `.fit`, `.fts` | [`FITSParser`][virtualizarr.parsers.FITSParser] | `virtualizarr[fits]` | ✅ | `numcodecs` for scaled (`BSCALE`/`BZERO`) images. `virtualizarr` for ASCII tables. | <span class="fmt-tag fmt-physics">🔭 astronomy</span> | Image/cube HDUs and ASCII tables are supported. Binary tables (`BINTABLE`) are not yet supported. |
| GeoTIFF | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-earth">🗺️ geospatial</span> <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> | TIFF. Also readable with [`OversightMLParser`](https://awslabs.github.io/osml-imagery-io/api/virtualizarr-parsers.html) from osml-imagery-io. |
| GRIB1 / GRIB2 | `.grib`, `.grb`, `.grib2`, `.grb2` | [`GribberishParser`](api/parsers/grib.md) | [gribberish](https://github.com/mpiannucci/gribberish) (`virtualizarr[grib]`) | ✅ | `gribberish` | <span class="fmt-tag fmt-water">🌦️ weather & climate</span> | Covers NWP model output such as HRRR and NBM. Each GRIB message becomes one chunk, decoded by a gribberish codec. GRIB1 complex and spherical-harmonic packing are not yet supported. Files whose messages don't form one hypercube need `open_virtual_datatree`. |
| HDF-EOS2 | `.hdf` | [`HDF4Parser`][virtualizarr.parsers.HDF4Parser] | `virtualizarr[hdf4]` | ✅ | `numcodecs` | <span class="fmt-tag fmt-earth">🌍 earth observation</span> | HDF4 underneath, e.g. older MODIS products. |
| HDF-EOS5 | `.he5`, `.h5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-earth">🌍 earth observation</span> | HDF5 underneath, e.g. Aura and OMI products. |
| HDF4 | `.hdf`, `.h4`, `.hdf4` | [`HDF4Parser`][virtualizarr.parsers.HDF4Parser] | `virtualizarr[hdf4]` | ✅ | `numcodecs` | <span class="fmt-tag fmt-earth">🌍 earth observation</span> | Uses kerchunk internally. |
| HDF5 | `.h5`, `.hdf5`, `.he5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-data">📦 general</span> | HDF5 links and object references are opt-in: see [HDF5 links and references](how_to/hdf5_links_references.md). |
| HEC-RAS | `.hdf` (`.p##.hdf`, `.g##.hdf`) | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-water">💧 hydrology</span> | HDF5 underneath. The HDF5 contents can be virtualized, but the model-specific mesh structure isn't interpreted. |
| Icechunk | (a repository, not a file) | [`IcechunkParser`][virtualizarr.parsers.IcechunkParser] | `virtualizarr[icechunk]` | ✅ | Same as the source data | <span class="fmt-tag fmt-data">📦 general</span> | Re-virtualizes an existing Icechunk repository without copying data. Requires `icechunk>=2.0.5`. |
| Imaris | `.ims` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-life">🔬 microscopy</span> | HDF5 underneath. |
| JPEG 2000 | `.jp2`, `.j2k`, `.j2c` | [`OversightMLParser`](https://awslabs.github.io/osml-imagery-io/api/virtualizarr-parsers.html) | [osml-imagery-io](https://github.com/awslabs/osml-imagery-io) (`osml-imagery-io[virtualizarr]`) | ❌ | `osml-imagery-io` | <span class="fmt-tag fmt-earth">🗺️ geospatial</span> <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> | **Do not use this parser with Icechunk on JPEG 2000 files whose tiles are split into several pieces (tile-parts)** ([osml-imagery-io#18](https://github.com/awslabs/osml-imagery-io/issues/18)). |
| Kerchunk references (JSON) | `.json` | [`KerchunkJSONParser`][virtualizarr.parsers.KerchunkJSONParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag fmt-data">📦 general</span> | Reads references you already made with kerchunk, so you can convert them to Icechunk. |
| Kerchunk references (Parquet) | `.parquet` (directory) | [`KerchunkParquetParser`][virtualizarr.parsers.KerchunkParquetParser] | `virtualizarr[kerchunk_parquet]` | ✅ | Same as the source data | <span class="fmt-tag fmt-data">📦 general</span> | As above. |
| MATLAB v7.3 | `.mat` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-data">📦 general</span> | Only v7.3 `.mat` files are HDF5. Older `.mat` versions are a different format and aren't supported ([#1122](https://github.com/zarr-developers/VirtualiZarr/issues/1122)). |
| NetCDF3 (classic, 64-bit offset, CDF-5) | `.nc`, `.cdf` | [`NetCDF3Parser`][virtualizarr.parsers.NetCDF3Parser] | `virtualizarr` | ✅ | None | <span class="fmt-tag fmt-water">🌦️ weather & climate</span> <span class="fmt-tag fmt-water">🌊 ocean</span> | All variables are uncompressed. Record (unlimited-dimension) variables are interleaved on disk, so each record becomes its own chunk (chunks covering several records would need several byte ranges: [#1156](https://github.com/zarr-developers/VirtualiZarr/issues/1156)). |
| NetCDF4 | `.nc`, `.nc4` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-water">🌦️ weather & climate</span> <span class="fmt-tag fmt-water">🌊 ocean</span> | HDF5 underneath. Check which one you have with `ncdump -k`: `netCDF-4` means HDF5, `classic` means NetCDF3. |
| NeXus | `.nxs`, `.h5` | [`HDFParser`][virtualizarr.parsers.HDFParser] | `virtualizarr[hdf]` | ✅ | `numcodecs`, plus `imagecodecs` for HDF5 plugin filters (e.g. LZF, LZ4, bzip2) | <span class="fmt-tag fmt-physics">⚛️ neutron & x-ray science</span> | HDF5 underneath. |
| NITF / NSIF (including SICD and SIDD SAR) | `.ntf`, `.nitf`, `.nsf` | [`OversightMLParser`](https://awslabs.github.io/osml-imagery-io/api/virtualizarr-parsers.html) | [osml-imagery-io](https://github.com/awslabs/osml-imagery-io) (`osml-imagery-io[virtualizarr]`) | ❌ | `osml-imagery-io` | <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> <span class="fmt-tag fmt-defence">🛡️ defence & intelligence</span> | NITF 2.1 and NSIF 1.0: all interleave modes, uncompressed, JPEG 2000, HTJ2K and JPEG DCT compression. Multi-file R-set pyramids become one group per resolution level. Same split-tile warning as JPEG 2000 ([osml-imagery-io#18](https://github.com/awslabs/osml-imagery-io/issues/18)). |
| OME-TIFF | `.ome.tif`, `.ome.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-life">🔬 microscopy</span> | TIFF. |
| OME-Zarr | `.ome.zarr`, `.zarr` | [`ZarrParser`][virtualizarr.parsers.ZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag fmt-life">🔬 microscopy</span> <span class="fmt-tag fmt-life">🦠 bioimaging</span> | Zarr. |
| TIFF | `.tif`, `.tiff` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-data">📦 general</span> | Internally tiled TIFFs work well, even if they aren't COG-compliant. Also readable with [`OversightMLParser`](https://awslabs.github.io/osml-imagery-io/api/virtualizarr-parsers.html) from osml-imagery-io. |
| Whole-slide images (e.g. Aperio) | `.svs`, `.tif` | [`VirtualTIFF`](https://virtual-tiff.readthedocs.io/en/latest/api/parser/) | [virtual-tiff](https://github.com/virtual-zarr/virtual-tiff) (`virtualizarr[tiff]`) | ✅ | `virtual-tiff`, plus `imagecodecs` for some compressions | <span class="fmt-tag fmt-life">🧫 pathology</span> <span class="fmt-tag fmt-life">🔬 microscopy</span> | TIFF-based variants only. |
| Zarr V2 / V3 | `.zarr` (directory) | [`ZarrParser`][virtualizarr.parsers.ZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag fmt-data">📦 general</span> | Creates a "Zarr of Zarrs", useful for combining many existing stores into one. |
| Zipped Zarr | `.zarr.zip`, `.zip` | [`ZippedZarrParser`][virtualizarr.parsers.ZippedZarrParser] | `virtualizarr` | ✅ | Same as the source data | <span class="fmt-tag fmt-data">📦 general</span> | Reads the archive's central directory, so the zip is never extracted. Entries must be stored uncompressed (`ZIP_STORED`). |

## 🛠️ Formats without a parser yet

These formats could be virtualized today, but nobody has written a parser for them yet.
Contributions are welcome: comment on the linked issue, or [write your own parser](explanation/custom_parsers.md).

| Format | Extensions | Domains | Notes |
| --- | --- | --- | --- |
| DICOM | `.dcm` | <span class="fmt-tag fmt-life">🩻 medical imaging</span> | Widespread, but complicated ([#218](https://github.com/zarr-developers/VirtualiZarr/issues/218)). A frame split into several fragments would need a codec that strips the item tags between them. |
| Hugging Face safetensors | `.safetensors` | <span class="fmt-tag fmt-data">🤖 machine learning</span> | Uncompressed, with all metadata at the start of the file ([#367](https://github.com/zarr-developers/VirtualiZarr/issues/367)). |
| MATLAB (pre-v7.3) | `.mat` | <span class="fmt-tag fmt-data">📦 general</span> | See [#1122](https://github.com/zarr-developers/VirtualiZarr/issues/1122). For v7.3 files, use `HDFParser` (see above). |
| NASA CDF | `.cdf` | <span class="fmt-tag fmt-physics">☀️ space physics</span> | Not to be confused with netCDF. |
| NIfTI | `.nii` | <span class="fmt-tag fmt-life">🩻 medical imaging</span> | Uncompressed `.nii` should be simple. `.nii.gz` is gzipped as a whole file (see below). |
| NumPy | `.npy`, `.npz` | <span class="fmt-tag fmt-data">📦 general</span> | `.npy` is a header followed by raw bytes. `.npz` is a zip of `.npy` files. |
| Open-Meteo OM files | `.om` | <span class="fmt-tag fmt-water">🌦️ weather & climate</span> | See [open-meteo/om-file-format#24](https://github.com/open-meteo/om-file-format/issues/24). |
| PDS4 | `.xml` with `.img`, `.dat` | <span class="fmt-tag fmt-physics">🪐 planetary science</span> | An XML label describing raw binary data. |
| RPN / FST | `.fst` | <span class="fmt-tag fmt-water">🌦️ weather & climate</span> | Environment and Climate Change Canada's model output format. |
| SAFE | `.SAFE` (directory) | <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> | Copernicus Sentinel product packaging. The JPEG 2000 and GeoTIFF files inside each have a parser, but nothing yet reads a product as a whole. |

## 🚧 Formats we can't support yet

These formats need new features in VirtualiZarr and Icechunk before they can be virtualized.
Today a virtual chunk reference is one byte range, decoded by codecs configured once for the whole array.
These formats need chunks made of several byte ranges ([#1156](https://github.com/zarr-developers/VirtualiZarr/issues/1156)), codec parameters that differ from chunk to chunk, or chunks of different sizes along one axis.

| Format | Extensions | Domains | Problem | Feature needed |
| --- | --- | --- | --- | --- |
| Compressed audio (AAC, MP3) | `.m4a`, `.aac`, `.mp3` | <span class="fmt-tag fmt-media">🔊 audio</span> <span class="fmt-tag fmt-media">🐦 bioacoustics</span> | Decoding a chunk needs lead-in frames from the previous chunk (AAC's overlapping windows, MP3's bit reservoir). | Per-chunk codec parameters: how many lead-in samples to drop, and frame sizes in MP4. |
| Compressed video (H.264, HEVC, AV1, VP9 in MP4 or QuickTime) | `.mp4`, `.mov`, `.m4v` | <span class="fmt-tag fmt-media">🎬 video</span> | Most frames are predicted from earlier ones, so a chunk has to be decoded from the previous keyframe, and its frames may need reordering. Decoding is also expensive. | Per-chunk codec parameters: frame sizes, display order, and how many frames to drop. |
| JPEG 2000 with split tiles | `.jp2`, `.j2k`, `.ntf` | <span class="fmt-tag fmt-earth">🗺️ geospatial</span> <span class="fmt-tag fmt-earth">🛰️ remote sensing</span> | The encoder can split each tile into pieces (often one per resolution level) and write the coarse pieces of every tile first, then the finer ones. Decoding a tile needs all its pieces, spread across the file. Don't use `OversightMLParser` with these files and Icechunk ([osml-imagery-io#18](https://github.com/awslabs/osml-imagery-io/issues/18)). | Chunks made of several byte ranges ([#1156](https://github.com/zarr-developers/VirtualiZarr/issues/1156)). |
| Parquet | `.parquet` | <span class="fmt-tag fmt-data">📊 tabular</span> | A plain-encoded, fixed-width, non-null column could almost be read today: each data page is a compressed run of raw values. But pages are cut by byte size, so they hold different numbers of rows, and most writers dictionary-encode by default, which puts the dictionary in a separate page ([#441](https://github.com/zarr-developers/VirtualiZarr/issues/441)). | Chunks of different sizes along an axis (a rectilinear chunk grid). For dictionary-encoded columns, also chunks made of several byte ranges or per-chunk codec parameters. |

## ⛔ Formats we can never support

These formats can't be virtualized even with the new kinds of chunk reference described above.
Reading a chunk would need most of the file, thousands of tiny byte ranges, expensive parsing, or computation that combines several sources.
See [Can my file format be virtualized?](explanation/faq.md#can-my-file-format-be-virtualized) for the general rules.

| Format | Domains | Why not |
| --- | --- | --- |
| Any whole-file-compressed file (e.g. `.nc.gz`, `.h5.bz2`, `.nii.gz`) | <span class="fmt-tag fmt-data">📦 general</span> | Compressing the whole file mixes up the bytes of every chunk. Decompress first, or (for small files) treat the whole file as one chunk. |
| CSV | <span class="fmt-tag fmt-data">📦 general</span> | A text format: every read needs expensive parsing, and rows don't map onto fixed-size chunks ([#200](https://github.com/zarr-developers/VirtualiZarr/issues/200)). |
| Furuno, Sigmet/IRIS, NEXRAD Level II | <span class="fmt-tag fmt-water">📡 weather radar</span> | Variables are interleaved in small blocks within each record, so one chunk would be thousands of tiny byte ranges ([#218](https://github.com/zarr-developers/VirtualiZarr/issues/218)). |
| GDAL Virtual Raster (`.vrt`) | <span class="fmt-tag fmt-earth">🗺️ geospatial</span> | A VRT can resample, reproject and mosaic its source rasters on the fly, which plain byte-range references can't express ([#166](https://github.com/zarr-developers/VirtualiZarr/issues/166)). Virtualize the underlying GeoTIFFs instead. |
| VCF | <span class="fmt-tag fmt-life">🧬 genomics</span> | A text format with interleaved columns. Convert it to Zarr with [bio2zarr](https://github.com/sgkit-dev/bio2zarr) instead. |
