import base64
import json
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import ujson
from numcodecs.abc import Codec
from xarray import Dataset, Variable
from xarray.backends.zarr import FillValueCoder, encode_zarr_variable
from xarray.coding.times import CFDatetimeCoder
from xarray.conventions import encode_dataset_coordinates
from zarr.core.buffer import default_buffer_prototype
from zarr.core.common import JSON
from zarr.core.metadata import ArrayV3Metadata
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.dtype import parse_data_type

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.manifest import join
from virtualizarr.manifests.utils import create_v3_array_metadata
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


def to_kerchunk_json_v3(metadata: ArrayV3Metadata) -> str:
    """Convert V3 metadata to the contents of a ``zarr.json`` kerchunk reference."""
    buffer = metadata.to_buffer_dict(default_buffer_prototype())["zarr.json"]
    # re-serialize to match the compact form used for the other references
    return json.dumps(json.loads(buffer.to_bytes()), separators=(",", ":"))


def _json_safe(attrs: dict[str, Any]) -> dict[str, JSON]:
    """Convert numpy scalars and arrays in attributes to their JSON equivalents."""
    return json.loads(json.dumps(attrs, cls=NumpyEncoder))


def dataset_to_kerchunk_refs(
    ds: Dataset, zarr_format: Literal[2, 3] = 2
) -> KerchunkStoreRefs:
    """
    Create a dictionary containing kerchunk-style store references from a single xarray.Dataset (which wraps ManifestArray objects).

    Parameters
    ----------
    ds
        The dataset to serialize.
    zarr_format
        The Zarr format of the metadata in the references. Format 2 writes ``.zgroup``,
        ``.zarray`` and ``.zattrs`` keys, and format 3 writes ``zarr.json`` keys.
    """
    if zarr_format not in (2, 3):
        raise ValueError(f"zarr_format must be 2 or 3, but got {zarr_format}")

    # xarray's .to_zarr() does this, so we need to do it for kerchunk too
    variables, attrs = encode_dataset_coordinates(ds)

    all_arr_refs = {}
    for var_name, var in variables.items():
        arr_refs = variable_to_kerchunk_arr_refs(
            var, str(var_name), zarr_format=zarr_format
        )

        prepended_with_var_name = {
            f"{var_name}/{key}": val for key, val in arr_refs.items()
        }
        all_arr_refs.update(prepended_with_var_name)

    group_refs: dict[str, Any]
    if zarr_format == 3:
        group_metadata = {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": _json_safe(attrs),
        }
        group_refs = {"zarr.json": json.dumps(group_metadata, separators=(",", ":"))}
    else:
        group_refs = {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": ujson.dumps(attrs),
        }

    ds_refs = {
        "version": 1,
        "refs": {
            **group_refs,
            **all_arr_refs,
        },
    }

    return cast(KerchunkStoreRefs, ds_refs)


def remove_file_uri_prefix(path: str):
    if path.startswith("file:///"):
        return path.removeprefix("file://")
    else:
        return path


def variable_to_kerchunk_arr_refs(
    var: Variable, var_name: str, zarr_format: Literal[2, 3] = 2
) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps either a ManifestArray or a numpy array).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).

    With ``zarr_format=3`` the array's metadata is one ``zarr.json`` reference, which
    holds the attributes and dimension names too, and the chunk keys follow the array's
    chunk key encoding (``c/0/0`` by default).
    """
    array_v3_metadata: ArrayV3Metadata | None = None

    if isinstance(var.data, ManifestArray):
        marr = var.data

        arr_refs: dict[str, str | list[str | int]] = {}
        for chunk_key, entry in marr.manifest.dict().items():
            if zarr_format == 3:
                indices = (
                    tuple(int(i) for i in chunk_key.split(".")) if marr.ndim else ()
                )
                ref_key = marr.metadata.encode_chunk_key(indices)
            else:
                ref_key = str(chunk_key)
            if "data" in entry:
                # Inlined chunk: emit as kerchunk's `base64:<b64>` form.
                arr_refs[ref_key] = (
                    b"base64:" + base64.b64encode(entry["data"])
                ).decode("utf-8")
            else:
                arr_refs[ref_key] = [
                    remove_file_uri_prefix(entry["path"]),
                    entry["offset"],
                    entry["length"],
                ]
        zattrs = {**var.attrs, **var.encoding}
        if zarr_format == 3:
            array_v3_metadata = marr.metadata
        else:
            array_v2_metadata = convert_v3_to_v2_metadata(marr.metadata)
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

        zattrs = {**var.attrs}
        if zarr_format == 3:
            # xarray reads the _FillValue attribute of a Zarr format 3 array in its encoded form
            if zattrs.get("_FillValue") is not None:
                zattrs["_FillValue"] = FillValueCoder.encode(
                    zattrs["_FillValue"], np_arr.dtype
                )
            # the whole array is one chunk; a zero-length axis still needs a chunk length of 1
            array_v3_metadata = create_v3_array_metadata(
                shape=np_arr.shape,
                data_type=np_arr.dtype,
                chunk_shape=tuple(max(n, 1) for n in np_arr.shape),
            )
            arr_refs = {
                array_v3_metadata.encode_chunk_key(
                    tuple(0 for _ in np_arr.shape)
                ): inlined_data
            }
        else:
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

    if array_v3_metadata is not None:
        array_v3_metadata = replace(
            array_v3_metadata,
            attributes=_json_safe(zattrs),
            dimension_names=tuple(str(dim) for dim in var.dims),
        )
        arr_refs["zarr.json"] = to_kerchunk_json_v3(array_v3_metadata)
    else:
        zarray_dict = to_kerchunk_json(array_v2_metadata)
        arr_refs[".zarray"] = zarray_dict

        zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
        arr_refs[".zattrs"] = json.dumps(
            zattrs, separators=(",", ":"), cls=NumpyEncoder
        )

    return cast(KerchunkArrRefs, arr_refs)
