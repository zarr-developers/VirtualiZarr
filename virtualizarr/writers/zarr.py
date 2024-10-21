from pathlib import Path

import numpy as np
from xarray import Dataset
from xarray.core.variable import Variable

from virtualizarr.vendor.zarr.utils import json_dumps
from virtualizarr.zarr import ZArray


def dataset_to_zarr(ds: Dataset, storepath: str) -> None:
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
        array_dir = _storepath / str(name)
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


def to_zarr_json(var: Variable, array_dir: Path) -> None:
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

    metadata = zarr_v3_array_metadata(
        marr.zarray, [str(x) for x in var.dims], var.attrs
    )
    with open(array_dir / "zarr.json", "wb") as metadata_file:
        metadata_file.write(json_dumps(metadata))


def zarr_v3_array_metadata(zarray: ZArray, dim_names: list[str], attrs: dict) -> dict:
    """Construct a v3-compliant metadata dict from v2 zarray + information stored on the xarray variable."""
    # TODO it would be nice if we could use the zarr-python metadata.ArrayMetadata classes to do this conversion for us
    metadata = zarray.dict()

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
    metadata["codecs"] = tuple(c.to_dict() for c in zarray._v3_codec_pipeline())
    metadata.pop("filters")
    metadata.pop("compressor")
    metadata.pop("order")

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
