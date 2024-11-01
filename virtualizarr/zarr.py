import dataclasses
from typing import TYPE_CHECKING, Any, Literal, NewType, cast

import numpy as np

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
        dtype = np.dtype(decoded_arr_refs_zarray["dtype"])
        fill_value = decoded_arr_refs_zarray["fill_value"]
        if np.issubdtype(dtype, np.floating) and (
            fill_value is None or fill_value == "NaN" or fill_value == "nan"
        ):
            fill_value = np.nan

        compressor = decoded_arr_refs_zarray["compressor"]
        zarr_format = int(decoded_arr_refs_zarray["zarr_format"])
        if zarr_format not in (2, 3):
            raise ValueError(f"Zarr format must be 2 or 3, but got {zarr_format}")

        return ZArray(
            chunks=tuple(decoded_arr_refs_zarray["chunks"]),
            compressor=compressor,
            dtype=dtype,
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
        import ujson

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

    def _v3_codec_pipeline(self) -> Any:
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
        try:
            from zarr.core.metadata.v3 import (  # type: ignore[import-untyped]
                parse_codecs,
            )
        except ImportError:
            raise ImportError("zarr v3 is required to generate v3 codec pipelines")

        codec_configs = []

        # https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/v1.0.html#transpose-codec-v1
        # Either "C" or "F", defining the layout of bytes within each chunk of the array.
        # "C" means row-major order, i.e., the last dimension varies fastest;
        # "F" means column-major order, i.e., the first dimension varies fastest.
        # For now, we only need transpose if the order is not "C"
        if self.order == "F":
            order = tuple(reversed(range(len(self.shape))))
            transpose = dict(name="transpose", configuration=dict(order=order))
            codec_configs.append(transpose)

        # https://github.com/zarr-developers/zarr-python/pull/1944#issuecomment-2151994097
        # "If no ArrayBytesCodec is supplied, we can auto-add a BytesCodec"
        bytes = dict(
            name="bytes", configuration={}
        )  # TODO need to handle endianess configuration
        codec_configs.append(bytes)

        # Noting here that zarr v3 has very few codecs specificed in the official spec,
        # and that there are far more codecs in `numcodecs`. We take a gamble and assume
        # that the codec names and configuration are simply mapped into zarrv3 "configurables".
        if self.filters:
            codec_configs.extend(
                [_num_codec_config_to_configurable(filter) for filter in self.filters]
            )

        if self.compressor:
            codec_configs.append(_num_codec_config_to_configurable(self.compressor))

        # convert the pipeline repr into actual v3 codec objects
        codec_pipeline = parse_codecs(codec_configs)

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


def determine_chunk_grid_shape(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(ceildiv(length, chunksize) for length, chunksize in zip(shape, chunks))


def _num_codec_config_to_configurable(num_codec: dict) -> dict:
    """
    Convert a numcodecs codec into a zarr v3 configurable.
    """
    if num_codec["id"].startswith("numcodecs."):
        return num_codec

    num_codec_copy = num_codec.copy()
    name = "numcodecs." + num_codec_copy.pop("id")
    return {"name": name, "configuration": num_codec_copy}
