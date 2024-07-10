import json
from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    NewType,
)

import numpy as np
import ujson  # type: ignore
import xarray as xr
from pydantic import BaseModel
from zarr.array import ArrayMetadata, ArrayV2Metadata, ArrayV3Metadata

from virtualizarr.vendor.zarr.utils import json_dumps

if TYPE_CHECKING:
    pass

# TODO replace these with classes imported directly from Zarr? (i.e. Zarr Object Models)
ZAttrs = NewType(
    "ZAttrs", dict[str, Any]
)  # just the .zattrs (for one array or for the whole store/group)


class Codec(BaseModel):
    """
    ZarrayV2 codec definition.
    """

    compressor: str | dict[str, Any] | None = None
    """
    If it's a string, it's the compressor ID.
    If it's a dict, it's the full compressor configuration.
    """
    filters: list[dict] | None = None

    def __repr__(self) -> str:
        return f"Codec(compressor={self.compressor}, filters={self.filters})"


def to_kerchunk_json(array: ArrayMetadata) -> str:
    return ujson.dumps(array.to_dict())


def from_kerchunk_refs(decoded_arr_refs_zarray: dict[str, Any]) -> ArrayMetadata:
    # coerce type of fill_value as kerchunk can be inconsistent with this
    fill_value = decoded_arr_refs_zarray["fill_value"]
    if fill_value is None or fill_value == "NaN" or fill_value == "nan":
        fill_value = np.nan

    compressor = decoded_arr_refs_zarray["compressor"]
    # deal with an inconsistency in kerchunk's tiff_to_zarr function
    # TODO should this be moved to the point where we actually call tiff_to_zarr? Or ideally made consistent upstream.
    if compressor is not None and "id" in compressor:
        compressor = compressor["id"]
    # return ArrayV2Metadata.from_dict(decoded_arr_refs_zarray)

    if int(decoded_arr_refs_zarray["zarr_format"]) == 2:
        return ArrayV2Metadata(
            shape=tuple(decoded_arr_refs_zarray["shape"]),
            dtype=np.dtype(decoded_arr_refs_zarray["dtype"]),
            chunks=tuple(decoded_arr_refs_zarray["chunks"]),
            order=decoded_arr_refs_zarray["order"],
            fill_value=fill_value,
            compressor=compressor,
            filters=decoded_arr_refs_zarray["filters"],
        )
    else:
        raise ValueError("Only Zarr format 2 is supported.")


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
    group_metadata = {"zarr_format": 3, "node_type": "group", "attributes": ds.attrs}
    with open(_storepath / "zarr.json", "wb") as group_metadata_file:
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

    marr.manifest.to_zarr_json(array_dir / "manifest.json")

    metadata = zarr_v3_array_metadata(marr.zarray, list(var.dims), var.attrs)
    with open(array_dir / "zarr.json", "wb") as metadata_file:
        metadata_file.write(json_dumps(metadata))


def zarr_v3_array_metadata(
    zarray: ArrayMetadata, dim_names: list[str], attrs: dict
) -> dict:
    """Construct a v3-compliant metadata dict from v2 zarray + information stored on the xarray variable."""
    # TODO it would be nice if we could use the zarr-python metadata.ArrayMetadata classes to do this conversion for us

    match zarray:
        case ArrayV3Metadata() as v3_metadata:
            metadata = replace(
                v3_metadata, dimension_names=tuple(dim_names), attributes=attrs
            ).to_dict()
            metadata["data_type"] = str(metadata["data_type"])

            # Can remove when ZarrV3 metadata includes storage transformers
            # https://github.com/zarr-developers/zarr-python/issues/2009
            if "storage_transformers" not in metadata:
                metadata["storage_transformers"] = [
                    {
                        "name": "chunk-manifest-json",
                        "configuration": {"manifest": "./manifest.json"},
                    }
                ]
            return metadata

        case ArrayV2Metadata():
            metadata = zarray.to_dict()
            # adjust to match v3 spec
            metadata["zarr_format"] = 3
            metadata["node_type"] = "array"
            metadata["data_type"] = str(np.dtype(metadata.pop("dtype")))
            metadata["chunk_grid"] = {
                "name": "regular",
                "configuration": {"chunk_shape": metadata.pop("chunks")},
            }
            metadata["chunk_key_encoding"] = {
                "name": "default",
                "configuration": {"separator": "/"},
            }
            metadata["codecs"] = metadata.pop("filters")
            metadata.pop("compressor")  # TODO this should be entered in codecs somehow
            metadata.pop("order")  # TODO this should be replaced by a transpose codec

            # indicate that we're using the manifest storage transformer ZEP
            metadata["storage_transformers"] = [
                {
                    "name": "chunk-manifest-json",
                    "configuration": {"manifest": "./manifest.json"},
                }
            ]

            # add information from xarray object
            metadata["dimension_names"] = dim_names
            metadata["attributes"] = attrs

            return metadata
        case _:
            raise ValueError("Unknown array metadata type")


def attrs_from_zarr_group_json(filepath: Path) -> dict:
    with open(filepath) as metadata_file:
        attrs = json.load(metadata_file)
    return attrs["attributes"]


def metadata_from_zarr_json(filepath: Path) -> tuple[ArrayMetadata, list[str], dict]:
    with open(filepath) as metadata_file:
        metadata = json.load(metadata_file)

    if {
        "name": "chunk-manifest-json",
        "configuration": {
            "manifest": "./manifest.json",
        },
    } not in metadata.get("storage_transformers", []):
        raise ValueError(
            "Can only read byte ranges from Zarr v3 stores which implement the manifest storage transformer ZEP."
        )
    # Can remove when ZarrV3 metadata includes storage transformers
    # https://github.com/zarr-developers/zarr-python/issues/2009
    metadata.pop("storage_transformers")
    metadata = ArrayV3Metadata.from_dict(metadata)
    return metadata, metadata.dimension_names, metadata.attributes
