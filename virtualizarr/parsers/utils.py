from typing import (
    Tuple,
    Union,
)

import numpy as np
from xarray.backends.zarr import FillValueCoder

FillValueType = Union[
    int,
    float,
    bool,
    complex,
    str,
    np.integer,
    np.floating,
    np.bool_,
    np.complexfloating,
    bytes,  # For fixed-length string storage
    Tuple[bytes, int],  # Structured type
]

def encode_cf_fill_value(
    fill_value: Union[np.ndarray, np.generic],
    target_dtype: np.dtype,
) -> FillValueType:
    """
    Convert the _FillValue attribute into one properly encoded for the target dtype.

    Parameters
    ----------
    fill_value
        An ndarray or value.
    target_dtype
        The target dtype of the ManifestArray that will use the _FillValue
    """
    if isinstance(fill_value, (np.ndarray, np.generic)):
        if isinstance(fill_value, np.ndarray) and fill_value.size > 1:
            raise ValueError("Expected a scalar")
        fillvalue = fill_value.item()
    else:
        fillvalue = fill_value
    encoded_fillvalue = FillValueCoder.encode(fillvalue, target_dtype)
    return encoded_fillvalue
