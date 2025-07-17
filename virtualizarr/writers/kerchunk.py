import base64
import json
from typing import cast

import numpy as np
import ujson
from numcodecs.abc import Codec
from xarray import Dataset, Variable
from xarray.backends.zarr import encode_zarr_variable
from xarray.coding.times import CFDatetimeCoder
from xarray.conventions import encode_dataset_coordinates
from zarr.core.common import JSON
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.dtype import parse_data_type

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.types.kerchunk import KerchunkArrRefs, KerchunkStoreRefs
from virtualizarr.utils import convert_v3_to_v2_metadata


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles common scientific Python types found in attributes.

    This encoder converts various Python types to JSON-serializable formats:
    - NumPy arrays and scalars to Python lists and native types
    - NumPy dtypes to strings
    - Sets to lists
    - Other objects that implement __array__ to lists
    - Objects with to_dict method (like pandas objects)
    - Objects with __str__ method as fallback
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        elif hasattr(obj, "__array__"):
            return np.asarray(obj).tolist()  # Handle array-like objects
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()  # Handle objects with to_dict method

        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            if hasattr(obj, "__str__"):
                return str(obj)
            raise


def to_kerchunk_json(v2_metadata: ArrayV2Metadata) -> str:
    """Convert V2 metadata to kerchunk JSON format."""

    zarray_dict: dict[str, JSON] = v2_metadata.to_dict()
    if v2_metadata.filters:
        zarray_dict["filters"] = [
            # we could also cast to json, but get_config is intended for serialization
            codec.get_config()
            for codec in v2_metadata.filters
            if codec is not None
        ]  # type: ignore[assignment]
    if isinstance(compressor := v2_metadata.compressor, Codec):
        zarray_dict["compressor"] = compressor.get_config()

    return json.dumps(zarray_dict, separators=(",", ":"), cls=NumpyEncoder)


def dataset_to_kerchunk_refs(ds: Dataset) -> KerchunkStoreRefs:
    """
    Create a dictionary containing kerchunk-style store references from a single xarray.Dataset (which wraps ManifestArray objects).
    """

    # xarray's .to_zarr() does this, so we need to do it for kerchunk too
    variables, attrs = encode_dataset_coordinates(ds)

    all_arr_refs = {}
    for var_name, var in variables.items():
        arr_refs = variable_to_kerchunk_arr_refs(var, str(var_name))

        prepended_with_var_name = {
            f"{var_name}/{key}": val for key, val in arr_refs.items()
        }
        all_arr_refs.update(prepended_with_var_name)

    ds_refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": ujson.dumps(attrs),
            **all_arr_refs,
        },
    }

    return cast(KerchunkStoreRefs, ds_refs)


def remove_file_uri_prefix(path: str):
    if path.startswith("file:///"):
        return path.removeprefix("file://")
    else:
        return path


def variable_to_kerchunk_arr_refs(var: Variable, var_name: str) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps either a ManifestArray or a numpy array).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).
    """

    if isinstance(var.data, ManifestArray):
        marr = var.data

        arr_refs: dict[str, str | list[str | int]] = {
            str(chunk_key): [
                remove_file_uri_prefix(entry["path"]),
                entry["offset"],
                entry["length"],
            ]
            for chunk_key, entry in marr.manifest.dict().items()
        }
        array_v2_metadata = convert_v3_to_v2_metadata(marr.metadata)
        zattrs = {**var.attrs, **var.encoding}
    else:
        var = encode_zarr_variable(var)
        try:
            np_arr = var.to_numpy()
        except AttributeError as e:
            raise TypeError(
                f"Can only serialize wrapped arrays of type ManifestArray or numpy.ndarray, but got type {type(var.data)}"
            ) from e

        if var.encoding:
            if "scale_factor" in var.encoding:
                raise NotImplementedError(
                    f"Cannot serialize loaded variable {var_name}, as it is encoded with a scale_factor"
                )
            if "offset" in var.encoding:
                raise NotImplementedError(
                    f"Cannot serialize loaded variable {var_name}, as it is encoded with an offset"
                )
            if "calendar" in var.encoding:
                np_arr = CFDatetimeCoder().encode(var.copy(), name=var_name).values
                dtype = var.encoding.get("dtype", None)
                if dtype and np_arr.dtype != dtype:
                    np_arr = np.asarray(np_arr, dtype=dtype)

        # This encoding is what kerchunk does when it "inlines" data, see https://github.com/fsspec/kerchunk/blob/a0c4f3b828d37f6d07995925b324595af68c4a19/kerchunk/hdf.py#L472
        byte_data = np_arr.tobytes()
        # TODO do I really need to encode then decode like this?
        inlined_data = (b"base64:" + base64.b64encode(byte_data)).decode("utf-8")

        # TODO can this be generalized to save individual chunks of a dask array?
        # TODO will this fail for a scalar?
        arr_refs = {join(0 for _ in np_arr.shape): inlined_data}

        array_v2_metadata = ArrayV2Metadata(
            chunks=np_arr.shape,
            shape=np_arr.shape,
            dtype=parse_data_type(
                np_arr.dtype, zarr_format=2
            ),  # needed unless zarr-python fixes https://github.com/zarr-developers/zarr-python/issues/3253
            order="C",
            fill_value=None,
        )
        zattrs = {**var.attrs}

    zarray_dict = to_kerchunk_json(array_v2_metadata)
    arr_refs[".zarray"] = zarray_dict

    zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
    arr_refs[".zattrs"] = json.dumps(zattrs, separators=(",", ":"), cls=NumpyEncoder)

    return cast(KerchunkArrRefs, arr_refs)
