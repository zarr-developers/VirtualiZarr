import json
from typing import Any

import numpy as np
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.metadata.v3 import V3JsonEncoder


def _replace_special_floats(obj: object) -> Any:
    """Helper function to replace NaN/Inf/-Inf values with special strings

    Note: this cannot be done in the V3JsonEncoder because Python's `json.dumps` optimistically
    converts NaN/Inf values to special types outside of the encoding step.
    """
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    elif isinstance(obj, dict):
        # Recursively replace in dictionaries
        return {k: _replace_special_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively replace in lists
        return [_replace_special_floats(item) for item in obj]
    return obj


def dict_to_buffer_dict(input: dict, prototype: BufferPrototype) -> dict[str, Buffer]:
    # modified from ArrayV3Metadata.to_buffer_dict
    d = _replace_special_floats(input)
    return prototype.buffer.from_bytes(json.dumps(d, cls=V3JsonEncoder).encode())
