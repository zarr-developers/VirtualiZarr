import base64
import json
from typing import Any, cast

import numpy as np
import ujson
from numcodecs.abc import Codec
from xarray import Dataset, Variable
from xarray.backends.zarr import encode_zarr_variable
from xarray.coding.times import CFDatetimeCoder
from xarray.conventions import encode_dataset_coordinates
from zarr.codecs import CastValue, ScaleOffset
from zarr.core.common import JSON
from zarr.core.metadata.v2 import ArrayV2Metadata
from zarr.core.metadata.v3 import ArrayV3Metadata
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


def _revert_cf_codecs_to_attrs(
    metadata: ArrayV3Metadata,
) -> tuple[ArrayV3Metadata, dict[str, Any]]:
    """
    Undo CF scale/offset codec packing for the zarr-v2-based kerchunk format.

    The HDF parser expresses CF scale_factor/add_offset as the zarr v3
    ``scale_offset`` + ``cast_value`` codecs, which numcodecs — and therefore
    kerchunk — cannot represent. Revert that here: drop the two codecs, restore
    the stored integer dtype and fill value, and hand scale_factor / add_offset
    back as attributes so xarray's ``decode_cf`` reapplies them on read (the
    long-standing v2 behaviour). Returns the input unchanged when no CF codecs
    are present.
    """
    scale_offset = next(
        (c for c in metadata.codecs if isinstance(c, ScaleOffset)), None
    )
    cast_value = next((c for c in metadata.codecs if isinstance(c, CastValue)), None)
    if scale_offset is None or cast_value is None:
        return metadata, {}

    scale, offset = scale_offset.scale, scale_offset.offset
    storage_dtype = cast_value.dtype.to_native_dtype()

    cf_attrs: dict[str, Any] = {"scale_factor": 1.0 / scale}
    if offset != 0:
        cf_attrs["add_offset"] = offset

    # re-pack the decoded float fill value into the stored integer domain,
    # mirroring the ScaleOffset codec's encode of `(x - offset) * scale`
    packed_fill = int(round((float(metadata.fill_value) - offset) * scale))

    remaining_codecs = [
        codec
        for codec in metadata.to_dict()["codecs"]
        if codec.get("name") not in ("scale_offset", "cast_value", "bytes")
    ]
    reverted = create_v3_array_metadata(
        shape=metadata.shape,
        data_type=storage_dtype,
        chunk_shape=metadata.chunks,
        fill_value=packed_fill,
        codecs=remaining_codecs,
        attributes=dict(metadata.attributes),
        dimension_names=metadata.dimension_names,
    )
    return reverted, cf_attrs


def variable_to_kerchunk_arr_refs(var: Variable, var_name: str) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps either a ManifestArray or a numpy array).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).
    """

    if isinstance(var.data, ManifestArray):
        marr = var.data

        arr_refs: dict[str, str | list[str | int]] = {}
        for chunk_key, entry in marr.manifest.dict().items():
            if "data" in entry:
                # Inlined chunk: emit as kerchunk's `base64:<b64>` form.
                arr_refs[str(chunk_key)] = (
                    b"base64:" + base64.b64encode(entry["data"])
                ).decode("utf-8")
            else:
                arr_refs[str(chunk_key)] = [
                    remove_file_uri_prefix(entry["path"]),
                    entry["offset"],
                    entry["length"],
                ]
        reverted_metadata, cf_attrs = _revert_cf_codecs_to_attrs(marr.metadata)
        array_v2_metadata = convert_v3_to_v2_metadata(reverted_metadata)
        zattrs = {**cf_attrs, **var.attrs, **var.encoding}
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
