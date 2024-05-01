
from pathlib import Path
from typing import Any, Literal, NewType, Optional, Tuple, Union, List, Dict, TYPE_CHECKING
import json

import numpy as np
import ujson  # type: ignore
import xarray as xr
from pydantic import BaseModel, ConfigDict, field_validator
from virtualizarr.vendor.zarr.utils import json_dumps

if TYPE_CHECKING:
    pass

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

    # should techically loop over groups in a tree but a dataset corresponds to only one group
    group_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": ds.attrs
    }
    with open(_storepath / 'zarr.json', "wb") as group_metadata_file:
        group_metadata_file.write(json_dumps(group_metadata))

    for name, var in ds.variables.items():
        array_dir = _storepath / name
        marr = var.data

        # TODO move this check outside the writing loop so we don't write an incomplete store on failure?
        # TODO at some point this should be generalized to also write in-memory arrays as normal zarr chunks, see GH isse #62.
        if not isinstance(marr, ManifestArray):
            raise TypeError(
                "Only xarray objects wrapping ManifestArrays can be written to zarr using this method, "
                f"but variable {name} wraps an array of type {type(marr)}"
            )

        Path.mkdir(array_dir, exist_ok=False)

        # write the chunk references into a manifest.json file
        # and the array metadata into a zarr.json file
        to_zarr_json(var, array_dir)


def to_zarr_json(var: xr.Variable, array_dir: Path) -> None:
    """
    Write out both the zarr.json and manifest.json file into the given zarr array directory.

    Follows the Zarr v3 manifest storage transformer ZEP (see https://github.com/zarr-developers/zarr-specs/issues/287).

    Parameters
    ----------
    var : xr.Variable
        Must be wrapping a ManifestArray
    dirpath : str
        Zarr store array directory into which to write files.
    """

    marr = var.data

    marr.manifest.to_zarr_json(array_dir / 'manifest.json')

    metadata = zarr_v3_array_metadata(marr.zarray, list(var.dims), var.attrs)
    with open(array_dir / 'zarr.json', "wb") as metadata_file:
        metadata_file.write(json_dumps(metadata))


def zarr_v3_array_metadata(zarray: ZArray, dim_names: List[str], attrs: dict) -> dict:
    """Construct a v3-compliant metadata dict from v2 zarray + information stored on the xarray variable."""
    # TODO it would be nice if we could use the zarr-python metadata.ArrayMetadata classes to do this conversion for us

    metadata = zarray.dict()

    # adjust to match v3 spec
    metadata["zarr_format"] = 3
    metadata["node_type"] = "array"
    metadata["data_type"] = str(np.dtype(metadata.pop("dtype")))
    metadata["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": metadata.pop("chunks")}}
    metadata["chunk_key_encoding"] = {
        "name": "default",
        "configuration": {
            "separator": "/"
        }
    }
    metadata["codecs"] = metadata.pop("filters")
    metadata.pop("compressor")  # TODO this should be entered in codecs somehow
    metadata.pop("order")  # TODO this should be replaced by a transpose codec

    # indicate that we're using the manifest storage transformer ZEP
    metadata["storage_transformers"] = [
        {
            "name": "chunk-manifest-json",
            "configuration": {
                "manifest": "./manifest.json"
            }
        }
    ]

    # add information from xarray object
    metadata["dimension_names"] = dim_names
    metadata["attributes"] = attrs

    return metadata


def attrs_from_zarr_group_json(filepath: Path) -> dict:
    with open(filepath, "r") as metadata_file:
        attrs = json.load(metadata_file)
    return attrs["attributes"]


def metadata_from_zarr_json(filepath: Path) -> Tuple[ZArray, List[str], dict]:
    with open(filepath, "r") as metadata_file:
        metadata = json.load(metadata_file)

    if {
            "name": "chunk-manifest-json",
            "configuration": {
                "manifest": "./manifest.json",
            }
       } not in metadata.get("storage_transformers", []):
        raise ValueError("Can only read byte ranges from Zarr v3 stores which implement the manifest storage transformer ZEP.")

    attrs = metadata.pop("attributes")
    dim_names = metadata.pop("dimension_names")

    chunk_shape = metadata["chunk_grid"]["configuration"]["chunk_shape"]

    zarray = ZArray(
        chunks=metadata["chunk_grid"]["configuration"]["chunk_shape"],
        compressor=metadata["codecs"],
        dtype=np.dtype(metadata["data_type"]),
        fill_value=metadata["fill_value"],
        filters=metadata.get("filters", None),
        order="C",
        shape=chunk_shape,
        zarr_format=3,
    )

    return zarray, dim_names, attrs
