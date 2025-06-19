import numpy as np
from xarray.backends.zarr import FillValueCoder

FillValueType = (
    int
    | float
    | bool
    | complex
    | str
    | np.integer
    | np.floating
    | np.bool_
    | np.complexfloating
    | bytes  # For fixed-length string storage
    | tuple[bytes, int]  # Structured type
)


def encode_cf_fill_value(
    fill_value: np.ndarray | np.generic,
    target_dtype: np.dtype,
) -> FillValueType:
    """
    Convert a fill value into one properly encoded for a target dtype.

    Parameters
    ----------
    fill_value
        An ndarray or value.
    target_dtype
        The target dtype of the ManifestArray that will use `fill_value` as its fill value.
    """
    if isinstance(fill_value, (np.ndarray, np.generic)):
        if isinstance(fill_value, np.ndarray) and fill_value.size > 1:
            raise ValueError("Expected a scalar")
        fillvalue = fill_value.item()
    else:
        fillvalue = fill_value
    encoded_fillvalue = FillValueCoder.encode(fillvalue, target_dtype)
    return encoded_fillvalue
