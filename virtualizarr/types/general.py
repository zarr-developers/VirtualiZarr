from typing import TYPE_CHECKING, NewType, TypeVar

ChunkKey = NewType("ChunkKey", str)  # a string of the form '1.0.1' etc.

if TYPE_CHECKING:
    from xarray import DataArray, Dataset

T_Xarray = TypeVar("T_Xarray", "DataArray", "Dataset")
