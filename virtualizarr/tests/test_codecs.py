import numpy as np
import pytest
from zarr.codecs import BytesCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.registry import get_codec_class

from conftest import (
    ARRAYBYTES_CODEC,
    BLOSC_CODEC,
    DELTA_CODEC,
    ZLIB_CODEC,
)
from virtualizarr.codecs import (
    convert_to_codec_pipeline,
    extract_codecs,
    get_codec_config,
    get_codecs,
)


class TestGetCodecs:
    """Test the get_codecs function."""

    def test_manifest_array_zarr_v3_default(self, manifest_array):
        """Test get_codecs with ManifestArray using default v3 codec."""
        test_manifest_array = manifest_array(codecs=None)
        actual_codecs = get_codecs(test_manifest_array)
        expected_codecs = tuple([BytesCodec(endian="little")])
        assert actual_codecs == expected_codecs

    def test_manifest_array_with_codecs(self, manifest_array):
        """Test get_codecs with ManifestArray using multiple v3 codecs."""
        test_codecs = [DELTA_CODEC, ARRAYBYTES_CODEC, BLOSC_CODEC]
        manifest_array = manifest_array(codecs=test_codecs)
        actual_codecs = get_codecs(manifest_array)
        assert actual_codecs == tuple(
            [
                get_codec_class(codec["name"])(**codec["configuration"])
                for codec in test_codecs
            ]
        )

    def test_zarr_v3_default_codecs(self, zarr_array):
        """Test get_codecs with Zarr array using default v3 codec."""
        zarr_array = zarr_array()
        actual_codecs = get_codecs(zarr_array)
        assert isinstance(actual_codecs[0], BytesCodec)

    def test_zarr_v3_with_codecs(self, zarr_array):
        """Test get_codecs with Zarr array using multiple v3 codecs."""
        test_codecs = [DELTA_CODEC, ARRAYBYTES_CODEC, BLOSC_CODEC]
        zarr_array = zarr_array(codecs=test_codecs)
        actual_codecs = get_codecs(zarr_array)
        assert actual_codecs == tuple(
            [
                get_codec_class(codec["name"])(**codec["configuration"])
                for codec in test_codecs
            ]
        )

    def test_zarr_v2_error(self, zarr_array):
        """Test that using v2 format raises an error."""
        zarr_array = zarr_array(zarr_format=2)
        with pytest.raises(
            ValueError,
            match="Only zarr v3 format arrays are supported. Please convert your array to v3 format.",
        ):
            get_codecs(zarr_array)


class TestConvertToCodecPipeline:
    """Test the convert_to_codec_pipeline function."""

    @pytest.mark.parametrize(
        "input_codecs,expected_pipeline",
        [
            # Case 1: No codecs - should result in just BytesCodec
            (
                None,
                BatchedCodecPipeline(
                    array_array_codecs=(),
                    array_bytes_codec=BytesCodec(endian="little"),
                    bytes_bytes_codecs=(),
                    batch_size=1,
                ),
            ),
            # Case 2: Delta codec - should result in DeltaCodec + BytesCodec
            (
                [DELTA_CODEC],
                BatchedCodecPipeline(
                    array_array_codecs=(
                        get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC),  # type: ignore[arg-type]
                    ),
                    array_bytes_codec=BytesCodec(endian="little"),
                    bytes_bytes_codecs=(),
                    batch_size=1,
                ),
            ),
            # Case 3: Delta + Blosc + Zlib - should result in all codecs + BytesCodec
            (
                [DELTA_CODEC, BLOSC_CODEC, ZLIB_CODEC],
                BatchedCodecPipeline(
                    array_array_codecs=(
                        get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC),  # type: ignore[arg-type]
                    ),
                    array_bytes_codec=BytesCodec(endian="little"),
                    bytes_bytes_codecs=(
                        get_codec_class(key="blosc").from_dict(BLOSC_CODEC),  # type: ignore[arg-type]
                        get_codec_class("numcodecs.zlib").from_dict(ZLIB_CODEC),  # type: ignore[arg-type]
                    ),
                    batch_size=1,
                ),
            ),
        ],
    )
    def test_convert_to_codec_pipeline_scenarios(self, input_codecs, expected_pipeline):
        """Test different scenarios for convert_to_codec_pipeline function."""
        dtype = np.dtype("<i4")
        if input_codecs is not None:
            input_codecs = list(input_codecs)

        result = convert_to_codec_pipeline(dtype=dtype, codecs=input_codecs)
        assert result == expected_pipeline

    @pytest.mark.parametrize(
        "dtype,expected_endian",
        [
            # Plain dtypes: numpy reports the byte order on the dtype itself.
            (np.dtype(">i4"), "big"),
            (np.dtype("<i4"), "little"),
            # Structured dtypes: numpy reports byteorder "|" on the dtype whatever its
            # fields say, so the byte order has to be read off the fields.
            (np.dtype([("flux", ">f4"), ("ivar", ">f4")]), "big"),
            (np.dtype([("flux", "<f4"), ("ivar", "<f4")]), "little"),
            # Fields that carry no byte order (single-byte, strings) are not evidence of
            # little-endian storage -- they are ignored.
            (np.dtype([("flux", ">f4"), ("name", "S8")]), "big"),
            (np.dtype([("flux", ">f4"), ("flag", "i1")]), "big"),
            (np.dtype([("flag", "i1"), ("name", "S8")]), "little"),
            # A subarray field hides its byte order behind ``.base``.
            (np.dtype([("flux", ">f4", (3,))]), "big"),
            # A nested struct hides it another level down.
            (np.dtype([("coadd", [("flux", ">f4")])]), "big"),
        ],
    )
    def test_byte_order_is_read_from_struct_fields(self, dtype, expected_endian):
        """The bytes codec's endian must describe the *stored* byte order.

        A FITS binary table arrives as a big-endian structured dtype; picking the codec from
        ``dtype.byteorder`` (which is "|" for any struct) silently stored it as little-endian.
        """
        result = convert_to_codec_pipeline(dtype=dtype, codecs=None)
        assert result.array_bytes_codec == BytesCodec(endian=expected_endian)

    def test_mixed_byte_order_struct_raises(self):
        """Zarr v3 gives an array a single bytes codec, so one array cannot store both."""
        with pytest.raises(ValueError, match="single byte order"):
            convert_to_codec_pipeline(
                dtype=np.dtype([("big", ">f4"), ("little", "<f4")]), codecs=None
            )


class TestExtractCodecs:
    """Test the extract_codecs function."""

    def test_extract_codecs_with_all_types(self):
        """Test extract_codecs with all types of codecs."""
        arrayarray_codec = get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC)
        arraybytes_codec = BytesCodec(endian="little")
        bytesbytes_codec = get_codec_class("numcodecs.zlib").from_dict(ZLIB_CODEC)

        codecs = (arrayarray_codec, arraybytes_codec, bytesbytes_codec)
        result = extract_codecs(codecs)

        assert result == (
            (arrayarray_codec,),
            arraybytes_codec,
            (bytesbytes_codec,),
        )

    def test_extract_codecs_with_only_arrayarray(self):
        """Test extract_codecs with only ArrayArrayCodec."""
        arrayarray_codec = get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC)

        codecs = (arrayarray_codec,)
        result = extract_codecs(codecs)

        assert result == (
            (arrayarray_codec,),
            None,
            (),
        )

    def test_extract_codecs_with_only_arraybytes(self):
        """Test extract_codecs with only ArrayBytesCodec."""
        arraybytes_codec = BytesCodec(endian="little")

        codecs = (arraybytes_codec,)
        result = extract_codecs(codecs)

        assert result == (
            (),
            arraybytes_codec,
            (),
        )

    def test_extract_codecs_with_only_bytesbytes(self):
        """Test extract_codecs with only BytesBytesCodec."""
        bytesbytes_codec = get_codec_class("numcodecs.zlib").from_dict(ZLIB_CODEC)

        codecs = (bytesbytes_codec,)
        result = extract_codecs(codecs)

        assert result == (
            (),
            None,
            (bytesbytes_codec,),
        )

    def test_extract_codecs_with_empty_list(self):
        """Test extract_codecs with an empty list."""
        codecs = ()
        result = extract_codecs(codecs)

        assert result == (
            (),
            None,
            (),
        )

    def test_raise_on_non_zarr_codec(self) -> None:
        class CustomCodec:
            """Custom codec which does not subclass from any of ArrayArrayCodec, ArrayBytesCodec, BytesBytesCodec."""

            pass

        codecs = (CustomCodec(),)
        with pytest.raises(TypeError, match="All codecs must be valid zarr v3 codecs"):
            extract_codecs(codecs)  # type: ignore[arg-type]


class TestGetCodecConfig:
    """Test the get_codec_config function."""

    def test_codec_with_codec_config(self):
        """Test get_codec_config with a codec having codec_config attribute."""
        codec = get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC)
        expected_config = codec.codec_config
        actual_config = get_codec_config(codec)
        assert actual_config == expected_config

    def test_codec_with_to_dict(self):
        """Test get_codec_config with a codec having get_config method."""
        from zarr.codecs import BloscCodec

        codec = BloscCodec(typesize=4, clevel=5, shuffle="shuffle", cname="lz4")
        expected_config = codec.to_dict()
        actual_config = get_codec_config(codec)
        assert actual_config == expected_config

    def test_codec_with_get_config(self):
        """Test get_codec_config with a codec having to_dict method."""
        from numcodecs import FixedScaleOffset

        codec = FixedScaleOffset(offset=0, scale=1, dtype="<i4")
        expected_config = codec.get_config()
        actual_config = get_codec_config(codec)
        assert actual_config == expected_config

    def test_codec_with_no_config_methods(self):
        """Test get_codec_config with a codec having no config methods."""

        class DummyCodec:
            pass

        codec = DummyCodec()
        with pytest.raises(ValueError, match="Unable to parse codec configuration:"):
            get_codec_config(codec)
