import logging

import fsspec
import numcodecs
import numcodecs.abc
import numpy as np
import zarr
from fsspec.implementations.reference import LazyReferenceMapper

from virtualizarr.vendor.kerchunk.codecs import AsciiTableCodec, VarArrCodec

try:
    from astropy.io import fits  # noqa: F401
    from astropy.wcs import WCS  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    raise ImportError(
        "astropy is required for kerchunking FITS files. Please install with "
        "`pip/conda install astropy`."
    ) from None

logger = logging.getLogger("fits-to-zarr")


BITPIX2DTYPE = {
    8: "uint8",
    16: ">i2",
    32: ">i4",
    64: ">i8",
    -32: "float32",
    -64: "float64",
}  # always bigendian


def process_file(
    url,
    storage_options=None,
    extension=None,
    inline_threshold=100,
    primary_attr_to_group=False,
    out=None,
):
    """
    Create JSON references for a single FITS file as a zarr group

    Parameters
    ----------
    url: str
        Where the file is
    storage_options: dict
        How to load that file (passed to fsspec)
    extension: list(int | str) | int | str or None
        Which extensions to include. Can be ordinal integer(s), the extension name (str) or if None,
        uses the first data extension
    inline_threshold: int
        (not yet implemented)
    primary_attr_to_group: bool
        Whether the output top-level group contains the attributes of the primary extension
        (which often contains no data, just a general description)
    out: dict-like or None
        This allows you to supply an fsspec.implementations.reference.LazyReferenceMapper
        to write out parquet as the references get filled, or some other dictionary-like class
        to customise how references get stored


    Returns
    -------
    dict of the references
    """
    from astropy.io import fits

    storage_options = storage_options or {}
    out = out or {}
    g = zarr.open(out)

    with fsspec.open(url, mode="rb", **storage_options) as f:
        infile = fits.open(f, do_not_scale_image_data=True)
        if extension is None:
            found = False
            for i, hdu in enumerate(infile):
                if hdu.header.get("NAXIS", 0) > 0:
                    extension = [i]
                    found = True
                    break
            if not found:
                raise ValueError("No data extensions")
        elif isinstance(extension, (int, str)):
            extension = [extension]

        for ext in extension:
            hdu = infile[ext]
            hdu.header.__str__()  # causes fixing of invalid cards

            attrs = dict(hdu.header)
            kwargs = {}
            if hdu.is_image:
                # for images/cubes (i.e., ndarrays with simple type)
                nax = hdu.header["NAXIS"]
                shape = tuple(int(hdu.header[f"NAXIS{i}"]) for i in range(nax, 0, -1))
                dtype = BITPIX2DTYPE[hdu.header["BITPIX"]]
                length = np.dtype(dtype).itemsize
                for s in shape:
                    length *= s

                if "BSCALE" in hdu.header or "BZERO" in hdu.header:
                    kwargs["filters"] = [
                        numcodecs.FixedScaleOffset(
                            offset=float(hdu.header.get("BZERO", 0)),
                            scale=float(hdu.header.get("BSCALE", 1)),
                            astype=dtype,
                            dtype=float,
                        )
                    ]
                else:
                    kwargs["filters"] = None
            elif isinstance(hdu, fits.hdu.table.TableHDU):
                # ascii table
                spans = hdu.columns._spans
                outdtype = [
                    [name, hdu.columns[name].format.recformat]
                    for name in hdu.columns.names
                ]
                indtypes = [
                    [name, f"S{i + 1}"] for name, i in zip(hdu.columns.names, spans)
                ]
                nrows = int(hdu.header["NAXIS2"])
                shape = (nrows,)
                kwargs["filters"] = [AsciiTableCodec(indtypes, outdtype)]
                dtype = [tuple(d) for d in outdtype]
                length = (sum(spans) + len(spans)) * nrows
            elif isinstance(hdu, fits.hdu.table.BinTableHDU):
                # binary table
                dtype = hdu.columns.dtype.newbyteorder(">")  # always big endian
                nrows = int(hdu.header["NAXIS2"])
                shape = (nrows,)
                # if hdu.fileinfo()["datSpan"] > length
                if any(_.format.startswith(("P", "Q")) for _ in hdu.columns):
                    # contains var fields
                    length = hdu.fileinfo()["datSpan"]
                    dt2 = [
                        (name, "O")
                        if hdu.columns[name].format.startswith(("P", "Q"))
                        else (name, str(dtype[name].base))
                        + ((dtype[name].shape,) if dtype[name].shape else ())
                        for name in dtype.names
                    ]
                    types = {
                        name: hdu.columns[name].format[1]
                        for name in dtype.names
                        if hdu.columns[name].format.startswith(("P", "Q"))
                    }
                    kwargs["object_codec"] = VarArrCodec(
                        str(dtype), str(dt2), nrows, types
                    )
                    dtype = dt2
                else:
                    length = dtype.itemsize * nrows
                    kwargs["filters"] = None
            else:
                logger.warning(f"Skipping unknown extension type: {hdu}")
                continue
            # one chunk for whole thing.
            # TODO: we could sub-chunk on biggest dimension
            name = hdu.name or str(ext)
            arr = g.empty(
                name=name,
                dtype=dtype,
                shape=shape,
                chunks=shape,
                compression=None,
                **kwargs,
            )
            arr.attrs.update(
                {
                    k: str(v) if not isinstance(v, (int, float, str)) else v
                    for k, v in attrs.items()
                    if k != "COMMENT"
                }
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"][-len(shape) :]
            loc = hdu.fileinfo()["datLoc"]
            parts = ".".join(["0"] * len(shape))
            out[f"{name}/{parts}"] = [url, loc, length]
        if primary_attr_to_group:
            # copy attributes of primary extension to top-level group
            hdu = infile[0]
            hdu.header.__str__()
            g.attrs.update(
                {
                    k: str(v) if not isinstance(v, (int, float, str)) else v
                    for k, v in dict(hdu.header).items()
                    if k != "COMMENT"
                }
            )
    if isinstance(out, LazyReferenceMapper):
        out.flush()
    return out
