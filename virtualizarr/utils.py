from __future__ import annotations

import datetime
import io
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import cftime
    import fsspec.core
    import fsspec.spec
    import xarray as xr
    from numpy.typing import NDArray

    # See pangeo_forge_recipes.storage
    OpenFileType = Union[
        fsspec.core.OpenFile, fsspec.spec.AbstractBufferedFile, io.IOBase
    ]


def _fsspec_openfile_from_filepath(
    *,
    filepath: str,
    reader_options: Optional[dict] = {},
) -> OpenFileType:
    """Converts input filepath to fsspec openfile object.

    Parameters
    ----------
    filepath : str
        Input filepath
    reader_options : _type_, optional
        Dict containing kwargs to pass to file opener, by default {'storage_options':{'key':'', 'secret':'', 'anon':True}}

    Returns
    -------
    OpenFileType
        An open file-like object, specific to the protocol supplied in filepath.

    Raises
    ------
    NotImplementedError
        Raises a Not Implemented Error if filepath protocol is not supported.
    """

    import fsspec
    from upath import UPath

    universal_filepath = UPath(filepath)
    protocol = universal_filepath.protocol

    if protocol == "s3":
        protocol_defaults = {"key": "", "secret": "", "anon": True}
    else:
        protocol_defaults = {}

    if reader_options is None:
        reader_options = {}

    storage_options = reader_options.get("storage_options", {})  # type: ignore

    # using dict merge operator to add in defaults if keys are not specified
    storage_options = protocol_defaults | storage_options
    fpath = fsspec.filesystem(protocol, **storage_options).open(filepath)

    return fpath


def encode_cftime(var: xr.Variable) -> NDArray[np.int64 | np.float64]:
    """Encode time variable

    Parameters
    ----------
    var : xr.Variable
        Variable containing cftime.datetime values and encoding information in
        ``var.encoding`` or ``var.attributes,

    Returns
    -------
    NDArray[np.int64 | np.float64]
        Numpy array of values encoded according to the units and calendar specified
        by encoding on ``var``
    """
    import cftime

    calendar = var.attrs.get("calendar", var.encoding.get("calendar", "standard"))
    units = var.attrs.get("units", var.encoding["units"])

    return cftime.date2num(var.data, calendar=calendar, units=units).ravel()


def recode_cftime(var: xr.Variable) -> NDArray[cftime.datetime]:
    """Recode time variable from np.datetime to cf.datetime

    Parameters
    ----------
    var : xr.Variable
        Variable containing np.datetime64 values and encoding information in
        `var.encoding`` or ``var.attributes``

    Returns
    -------
    NDArray[cftime.datetime]
        Numpy array of cftime.datetime values
    """
    import cftime

    calendar = var.attrs.get("calendar", var.encoding.get("calendar", "standard"))
    units = var.attrs.get("units", var.encoding["units"])

    values = []
    for c in var.values:
        value = cftime.num2date(
            cftime.date2num(
                datetime.datetime.fromisoformat(str(c)),
                calendar=calendar,
                units=units,
            ),
            calendar=calendar,
            units=units,
        )
        values.append(value)
    return np.array(values)
