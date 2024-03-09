from typing import Any, Literal, NewType, Optional, Tuple, TypedDict

import numpy as np

ChunkKey = NewType("ChunkKey", str)  # a string of the form '1.0.1' etc.

# TODO replace these with classes imported directly from Zarr?
ZAttrs = NewType(
    "ZAttrs", dict[str, Any]
)  # just the .zattrs (for one array or for the whole store/group)


# TODO use a dataclass instead?
class ZArray(TypedDict):
    """Just the .zarray information"""

    chunks: Tuple[int, ...]
    compressor: Optional[str]
    dtype: np.dtype
    fill_value: Optional[float]  # float or int?
    filters: Optional[str]
    order: Literal["C"] | Literal["F"]
    shape: Tuple[int, ...]
    zarr_format: int

    @classmethod
    def from_kerchunk_refs(cls, decoded_arr_refs_zarray) -> "ZArray":
        # TODO should we be doing some type coercion on the 'fill_value' here?

        return cls(
            {
                "chunks": tuple(decoded_arr_refs_zarray["chunks"]),
                "compressor": decoded_arr_refs_zarray["compressor"],
                "dtype": np.dtype(decoded_arr_refs_zarray["dtype"]),
                "fill_value": decoded_arr_refs_zarray["fill_value"],
                "filters": decoded_arr_refs_zarray["filters"],
                "order": decoded_arr_refs_zarray["order"],
                "shape": tuple(decoded_arr_refs_zarray["shape"]),
                "zarr_format": int(decoded_arr_refs_zarray["zarr_format"]),
            }
        )
