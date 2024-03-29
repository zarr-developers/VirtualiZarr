from typing import Any, Literal, NewType, Optional, Tuple, Union, List, Dict

import numpy as np
import ujson  # type: ignore
from pydantic import BaseModel, ConfigDict, field_validator

# TODO replace these with classes imported directly from Zarr? (i.e. Zarr Object Models)
ZAttrs = NewType(
    "ZAttrs", dict[str, Any]
)  # just the .zattrs (for one array or for the whole store/group)


class Codec(BaseModel):
    compressor: Optional[str] = None
    filters: Optional[List[Dict]] = None

    def __repr__(self) -> str:
        return f"Codec(compressor={self.compressor}, filters={self.filters})"


class ZArray(BaseModel):
    """Just the .zarray information"""

    # TODO will this work for V3?

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # only here so pydantic doesn't complain about the numpy dtype field
    )

    chunks: Tuple[int, ...]
    compressor: Optional[str] = None
    dtype: np.dtype
    fill_value: Optional[float] = None  # float or int?
    filters: Optional[List[Dict]] = None
    order: Union[Literal["C"], Literal["F"]]
    shape: Tuple[int, ...]
    zarr_format: Union[Literal[2], Literal[3]] = 2

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, dtype) -> np.dtype:
        # Your custom validation logic here
        # Convert numpy.dtype to a format suitable for Pydantic
        return np.dtype(dtype)

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.chunks):
            raise ValueError(
                "Dimension mismatch between array shape and chunk shape. "
                f"Array shape {self.shape} has ndim={self.shape} but chunk shape {self.chunks} has ndim={len(self.chunks)}"
            )

    @property
    def codec(self) -> Codec:
        """For comparison against other arrays."""
        return Codec(compressor=self.compressor, filters=self.filters)

    @property
    def shape_chunk_grid(self) -> Tuple[int, ...]:
        """Shape of the chunk grid implied by the array shape and chunk shape."""
        chunk_grid_shape = tuple(
            ceildiv(array_side_length, chunk_length)
            for array_side_length, chunk_length in zip(self.shape, self.chunks)
        )
        return chunk_grid_shape

    def __repr__(self) -> str:
        return f"ZArray(shape={self.shape}, chunks={self.chunks}, dtype={self.dtype}, compressor={self.compressor}, filters={self.filters}, fill_value={self.fill_value})"

    @classmethod
    def from_kerchunk_refs(cls, decoded_arr_refs_zarray) -> "ZArray":
        # TODO should we be doing some type coercion on the 'fill_value' here?

        return ZArray(
            chunks=tuple(decoded_arr_refs_zarray["chunks"]),
            compressor=decoded_arr_refs_zarray["compressor"],
            dtype=np.dtype(decoded_arr_refs_zarray["dtype"]),
            fill_value=decoded_arr_refs_zarray["fill_value"],
            filters=decoded_arr_refs_zarray["filters"],
            order=decoded_arr_refs_zarray["order"],
            shape=tuple(decoded_arr_refs_zarray["shape"]),
            zarr_format=int(decoded_arr_refs_zarray["zarr_format"]),
        )

    def to_kerchunk_json(self) -> str:
        zarray_dict = dict(self)
        # TODO not sure if there is a better way to get the '<i4' style representation of the dtype out
        zarray_dict["dtype"] = zarray_dict["dtype"].descr[0][1]
        return ujson.dumps(zarray_dict)


def ceildiv(a: int, b: int) -> int:
    """
    Ceiling division operator for integers.

    See https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(a // -b)
