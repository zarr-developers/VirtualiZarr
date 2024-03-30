
import json
from pathlib import Path
from typing import Any, Literal, NewType, Optional, Tuple, Union, List, Dict

import numpy as np
import ujson  # type: ignore
import xarray as xr
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

    def dict(self) -> dict[str, Any]:
        zarray_dict = dict(self)
        zarray_dict["dtype"] = encode_dtype(zarray_dict["dtype"])
        return zarray_dict

    def to_kerchunk_json(self) -> str:
        return ujson.dumps(self.dict())


def encode_dtype(dtype: np.dtype) -> str:
    # TODO not sure if there is a better way to get the '<i4' style representation of the dtype out
    return dtype.descr[0][1]


def ceildiv(a: int, b: int) -> int:
    """
    Ceiling division operator for integers.

    See https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    """
    return -(a // -b)


def dataset_to_zarr(ds: xr.Dataset, storepath: str) -> None:
    """
    Write an xarray dataset whose variables wrap ManifestArrays to a v3 Zarr store, writing chunk references into manifest.json files.

    Currently requires all variables to be backed by ManifestArray objects.

    Not very useful until some implementation of a Zarr reader can actually read these manifest.json files.
    See https://github.com/zarr-developers/zarr-specs/issues/287

    Parameters
    ----------
    ds: xr.Dataset
    storepath: str
    """

    from virtualizarr.manifests import ManifestArray

    _storepath = Path(storepath)
    Path.mkdir(_storepath, exist_ok=False)

    # TODO should techically loop over groups in a tree but a dataset corresponds to only one group
    # TODO does this mean we need a group kwarg?

    consolidated_metadata: dict = {"metadata": {}}

    # write top-level .zattrs
    with open(_storepath / ".zattrs", "w") as json_file:
        json.dump(ds.attrs, json_file, indent=4, separators=(", ", ": "))
    consolidated_metadata[".zattrs"] = ds.attrs

    # write .zgroup
    with open(_storepath / ".zgroup", "w") as json_file:
        json.dump({"zarr_format": 2}, json_file, indent=4, separators=(", ", ": "))
    consolidated_metadata[".zgroup"] = {"zarr_format": 2}

    for name, var in ds.variables.items():
        array_dir = _storepath / name
        marr = var.data

        # TODO move this check outside the writing loop so we don't write an incomplete store on failure?
        if not isinstance(marr, ManifestArray):
            raise TypeError(
                "Only xarray objects wrapping ManifestArrays can be written to zarr using this method, "
                f"but variable {name} wraps an array of type {type(marr)}"
            )

        Path.mkdir(array_dir, exist_ok=False)

        # write the chunk references into a manifest.json file
        marr.manifest.to_zarr_json(array_dir / "manifest.json")

        # write each .zarray
        with open(array_dir / ".zarray", "w") as json_file:
            json.dump(marr.zarray.dict(), json_file, indent=4, separators=(", ", ": "))

        # write each .zattrs
        zattrs = var.attrs.copy()
        zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
        with open(array_dir / ".zattrs", "w") as json_file:
            json.dump(zattrs, json_file, indent=4, separators=(", ", ": "))

        # record this info to include in the overall .zmetadata
        consolidated_metadata["metadata"][name + "/.zarray"] = marr.zarray.dict()
        consolidated_metadata["metadata"][name + "/.zattrs"] = zattrs

    # write store-level .zmetadata
    consolidated_metadata["zarr_consolidated_format"] = 1
    with open(_storepath / ".zmetadata", "w") as json_file:
        json.dump(consolidated_metadata, json_file, indent=4, separators=(", ", ": "))
