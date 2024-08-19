import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NewType, cast

import numcodecs
import numpy as np
import ujson  # type: ignore
import xarray as xr

from virtualizarr.vendor.zarr.utils import json_dumps

if TYPE_CHECKING:
    pass

# TODO replace these with classes imported directly from Zarr? (i.e. Zarr Object Models)
ZAttrs = NewType(
    "ZAttrs", dict[str, Any]
)  # just the .zattrs (for one array or for the whole store/group)
FillValueT = bool | str | float | int | list | None
ZARR_FORMAT = Literal[2, 3]

ZARR_DEFAULT_FILL_VALUE: dict[str, FillValueT] = {
    # numpy dtypes's hierarchy lets us avoid checking for all the widths
    # https://numpy.org/doc/stable/reference/arrays.scalars.html
    np.dtype("bool").kind: False,
    np.dtype("int").kind: 0,
    np.dtype("float").kind: 0.0,
    np.dtype("complex").kind: [0.0, 0.0],
    np.dtype("datetime64").kind: 0,
}
"""
The value and format of the fill_value depend on the `data_type` of the array.
See here for spec:
https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value
"""


@dataclasses.dataclass
class Codec:
    compressor: dict | None = None
    filters: list[dict] | None = None


@dataclasses.dataclass
class ZArray:
    """Just the .zarray information"""

    # TODO will this work for V3?

    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    fill_value: FillValueT = dataclasses.field(default=None)
    order: Literal["C", "F"] = "C"
    compressor: dict | None = None
    filters: list[dict] | None = None
    zarr_format: Literal[2, 3] = 2

    def __post_init__(self) -> None:
        if len(self.shape) != len(self.chunks):
            raise ValueError(
                "Dimension mismatch between array shape and chunk shape. "
                f"Array shape {self.shape} has ndim={self.shape} but chunk shape {self.chunks} has ndim={len(self.chunks)}"
            )

        if isinstance(self.dtype, str):
            # Convert dtype string to numpy.dtype
            self.dtype = np.dtype(self.dtype)

        if self.fill_value is None:
            self.fill_value = ZARR_DEFAULT_FILL_VALUE.get(self.dtype.kind, 0.0)

    @property
    def codec(self) -> Codec:
        """For comparison against other arrays."""
        return Codec(compressor=self.compressor, filters=self.filters)

    @classmethod
    def from_kerchunk_refs(cls, decoded_arr_refs_zarray) -> "ZArray":
        # coerce type of fill_value as kerchunk can be inconsistent with this
        fill_value = decoded_arr_refs_zarray["fill_value"]
        if fill_value is None or fill_value == "NaN" or fill_value == "nan":
            fill_value = np.nan

        compressor = decoded_arr_refs_zarray["compressor"]
        zarr_format = int(decoded_arr_refs_zarray["zarr_format"])
        if zarr_format not in (2, 3):
            raise ValueError(f"Zarr format must be 2 or 3, but got {zarr_format}")

        return ZArray(
            chunks=tuple(decoded_arr_refs_zarray["chunks"]),
            compressor=compressor,
            dtype=np.dtype(decoded_arr_refs_zarray["dtype"]),
            fill_value=fill_value,
            filters=decoded_arr_refs_zarray["filters"],
            order=decoded_arr_refs_zarray["order"],
            shape=tuple(decoded_arr_refs_zarray["shape"]),
            zarr_format=cast(ZARR_FORMAT, zarr_format),
        )

    def dict(self) -> dict[str, Any]:
        zarray_dict = dataclasses.asdict(self)
        zarray_dict["dtype"] = encode_dtype(zarray_dict["dtype"])
        return zarray_dict

    def to_kerchunk_json(self) -> str:
        zarray_dict = self.dict()
        if zarray_dict["fill_value"] is np.nan:
            zarray_dict["fill_value"] = None
        return ujson.dumps(zarray_dict)

    # ZArray.dict seems to shadow "dict", so we need the type ignore in
    # the signature below.
    def replace(
        self,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | str | None = None,
        fill_value: FillValueT = None,
        order: Literal["C", "F"] | None = None,
        compressor: "dict | None" = None,  # type: ignore[valid-type]
        filters: list[dict] | None = None,  # type: ignore[valid-type]
        zarr_format: Literal[2, 3] | None = None,
    ) -> "ZArray":
        """
        Convenience method to create a new ZArray from an existing one by altering only certain attributes.
        """
        replacements: dict[str, Any] = {}
        if shape is not None:
            replacements["shape"] = shape
        if chunks is not None:
            replacements["chunks"] = chunks
        if dtype is not None:
            replacements["dtype"] = dtype
        if fill_value is not None:
            replacements["fill_value"] = fill_value
        if order is not None:
            replacements["order"] = order
        if compressor is not None:
            replacements["compressor"] = compressor
        if filters is not None:
            replacements["filters"] = filters
        if zarr_format is not None:
            replacements["zarr_format"] = zarr_format
        return dataclasses.replace(self, **replacements)

    def _v3_codec_pipeline(self) -> list:
        """
        VirtualiZarr internally uses the `filters`, `compressor`, and `order` attributes
        from zarr v2, but to create conformant zarr v3 metadata those 3 must be turned into `codecs` objects.
        Not all codecs are created equal though: https://github.com/zarr-developers/zarr-python/issues/1943
        An array _must_ declare a single ArrayBytes codec, and 0 or more ArrayArray, BytesBytes codecs.
        Roughly, this is the mapping:
        ```
            filters: Iterable[ArrayArrayCodec] #optional
            compressor: ArrayBytesCodec #mandatory
            post_compressor: Iterable[BytesBytesCodec] #optional
        ```
        """
        if self.filters:
            filter_codecs_configs = [
                numcodecs.get_codec(filter).get_config() for filter in self.filters
            ]
            filters = [
                dict(name=codec.pop("id"), configuration=codec)
                for codec in filter_codecs_configs
            ]
        else:
            filters = []

        # Noting here that zarr v3 has very few codecs specificed in the official spec,
        # and that there are far more codecs in `numcodecs`. We take a gamble and assume
        # that the codec names and configuration are simply mapped into zarrv3 "configurables".
        if self.compressor:
            compressor = [_num_codec_config_to_configurable(self.compressor)]
        else:
            compressor = []

        # https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/v1.0.html#transpose-codec-v1
        # Either "C" or "F", defining the layout of bytes within each chunk of the array.
        # "C" means row-major order, i.e., the last dimension varies fastest;
        # "F" means column-major order, i.e., the first dimension varies fastest.
        if self.order == "C":
            order = tuple(range(len(self.shape)))
        elif self.order == "F":
            order = tuple(reversed(range(len(self.shape))))

        transpose = dict(name="transpose", configuration=dict(order=order))
        # https://github.com/zarr-developers/zarr-python/pull/1944#issuecomment-2151994097
        # "If no ArrayBytesCodec is supplied, we can auto-add a BytesCodec"
        bytes = dict(
            name="bytes", configuration={}
        )  # TODO need to handle endianess configuration

        # The order here is significant!
        # [ArrayArray] -> ArrayBytes -> [BytesBytes]
        codec_pipeline = [transpose, bytes] + compressor + filters
        return codec_pipeline


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
    metadata["codecs"] = zarray._v3_codec_pipeline()
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


def attrs_from_zarr_group_json(filepath: Path) -> dict:
    with open(filepath) as metadata_file:
        attrs = json.load(metadata_file)
    return attrs["attributes"]


def metadata_from_zarr_json(filepath: Path) -> tuple[ZArray, list[str], dict]:
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

    attrs = metadata.pop("attributes")
    dim_names = metadata.pop("dimension_names")

    chunk_shape = tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
    shape = tuple(metadata["shape"])
    zarr_format = metadata["zarr_format"]

    if metadata["fill_value"] is None:
        raise ValueError(
            "fill_value must be specified https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#fill-value"
        )
    else:
        fill_value = metadata["fill_value"]

    all_codecs = [
        codec
        for codec in metadata["codecs"]
        if codec["name"] not in ("transpose", "bytes")
    ]
    compressor, *filters = [
        _configurable_to_num_codec_config(_filter) for _filter in all_codecs
    ]
    zarray = ZArray(
        chunks=chunk_shape,
        compressor=compressor,
        dtype=np.dtype(metadata["data_type"]),
        fill_value=fill_value,
        filters=filters or None,
        order="C",
        shape=shape,
        zarr_format=zarr_format,
    )

    return zarray, dim_names, attrs


def _configurable_to_num_codec_config(configurable: dict) -> dict:
    """
    Convert a zarr v3 configurable into a numcodecs codec.
    """
    configurable_copy = configurable.copy()
    codec_id = configurable_copy.pop("name")
    configuration = configurable_copy.pop("configuration")
    return numcodecs.get_codec({"id": codec_id, **configuration}).get_config()


def _num_codec_config_to_configurable(num_codec: dict) -> dict:
    """
    Convert a numcodecs codec into a zarr v3 configurable.
    """
    num_codec_copy = num_codec.copy()
    return {"name": num_codec_copy.pop("id"), "configuration": num_codec_copy}
