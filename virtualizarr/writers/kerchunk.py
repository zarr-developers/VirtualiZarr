import base64
import json
from typing import Any, cast

import numpy as np
from xarray import Dataset
from xarray.coding.times import CFDatetimeCoder
from xarray.core.variable import Variable
from zarr.abc.codec import ArrayArrayCodec, BytesBytesCodec
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from virtualizarr.codecs import extract_codecs, get_codec_config
from virtualizarr.manifests.manifest import join
from virtualizarr.types.kerchunk import KerchunkArrRefs, KerchunkStoreRefs


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


def dataset_to_kerchunk_refs(ds: Dataset) -> KerchunkStoreRefs:
    """
    Create a dictionary containing kerchunk-style store references from a single xarray.Dataset (which wraps ManifestArray objects).
    """

    import ujson

    all_arr_refs = {}
    for var_name, var in ds.variables.items():
        arr_refs = variable_to_kerchunk_arr_refs(var, str(var_name))

        prepended_with_var_name = {
            f"{var_name}/{key}": val for key, val in arr_refs.items()
        }
        all_arr_refs.update(prepended_with_var_name)

    zattrs = ds.attrs
    if ds.coords:
        coord_names = [str(x) for x in ds.coords]
        # this weird concatenated string instead of a list of strings is inconsistent with how other features in the kerchunk references format are stored
        # see https://github.com/zarr-developers/VirtualiZarr/issues/105#issuecomment-2187266739
        zattrs["coordinates"] = " ".join(coord_names)

    ds_refs = {
        "version": 1,
        "refs": {
            ".zgroup": '{"zarr_format":2}',
            ".zattrs": ujson.dumps(zattrs),
            **all_arr_refs,
        },
    }

    return cast(KerchunkStoreRefs, ds_refs)


def remove_file_uri_prefix(path: str):
    if path.startswith("file:///"):
        return path.removeprefix("file://")
    else:
        return path


def convert_v3_to_v2_metadata(
    v3_metadata: ArrayV3Metadata, fill_value: Any = None
) -> ArrayV2Metadata:
    """
    Convert ArrayV3Metadata to ArrayV2Metadata.

    Parameters
    ----------
    v3_metadata : ArrayV3Metadata
        The metadata object in v3 format.
    fill_value : Any, optional
        Override the fill value from v3 metadata.

    Returns
    -------
    ArrayV2Metadata
        The metadata object in v2 format.
    """
    import warnings

    array_filters: tuple[ArrayArrayCodec, ...]
    bytes_compressors: tuple[BytesBytesCodec, ...]
    array_filters, _, bytes_compressors = extract_codecs(v3_metadata.codecs)

    # Handle compressor configuration
    compressor_config: dict[str, Any] | None = None
    if bytes_compressors:
        if len(bytes_compressors) > 1:
            warnings.warn(
                "Multiple compressors found in v3 metadata. Using the first compressor, "
                "others will be ignored. This may affect data compatibility.",
                UserWarning,
            )
        compressor_config = get_codec_config(bytes_compressors[0])

    # Handle filter configurations
    filter_configs = [get_codec_config(filter_) for filter_ in array_filters]
    v2_metadata = ArrayV2Metadata(
        shape=v3_metadata.shape,
        dtype=v3_metadata.data_type.to_numpy(),
        chunks=v3_metadata.chunks,
        fill_value=fill_value or v3_metadata.fill_value,
        compressor=compressor_config,
        filters=filter_configs,
        order="C",
        attributes=v3_metadata.attributes,
        dimension_separator=".",  # Assuming '.' as default dimension separator
    )
    return v2_metadata


def variable_to_kerchunk_arr_refs(var: Variable, var_name: str) -> KerchunkArrRefs:
    """
    Create a dictionary containing kerchunk-style array references from a single xarray.Variable (which wraps either a ManifestArray or a numpy array).

    Partially encodes the inner dicts to json to match kerchunk behaviour (see https://github.com/fsspec/kerchunk/issues/415).
    """
    from virtualizarr.manifests import ManifestArray
    from virtualizarr.translators.kerchunk import to_kerchunk_json

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
    else:
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

        # This encoding is what kerchunk does when it "inlines" data, see https://github.com/fsspec/kerchunk/blob/a0c4f3b828d37f6d07995925b324595af68c4a19/kerchunk/hdf.py#L472
        byte_data = np_arr.tobytes()
        # TODO do I really need to encode then decode like this?
        inlined_data = (b"base64:" + base64.b64encode(byte_data)).decode("utf-8")

        # TODO can this be generalized to save individual chunks of a dask array?
        # TODO will this fail for a scalar?
        arr_refs = {join(0 for _ in np_arr.shape): inlined_data}

        from zarr.core.metadata.v2 import ArrayV2Metadata

        array_v2_metadata = ArrayV2Metadata(
            chunks=np_arr.shape,
            shape=np_arr.shape,
            dtype=np_arr.dtype,
            order="C",
            fill_value=var.encoding.get("fill_value", None),
        )

    zarray_dict = to_kerchunk_json(array_v2_metadata)
    arr_refs[".zarray"] = zarray_dict

    zattrs = {**var.attrs, **var.encoding}
    zattrs["_ARRAY_DIMENSIONS"] = list(var.dims)
    arr_refs[".zattrs"] = json.dumps(zattrs, separators=(",", ":"), cls=NumpyEncoder)

    return cast(KerchunkArrRefs, arr_refs)
