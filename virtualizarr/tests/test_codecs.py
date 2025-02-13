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
from virtualizarr.codecs import convert_to_codec_pipeline, get_codecs


class TestCodecs:
    def create_zarr_array(self, codecs=None, zarr_format=3):
        """Create a test Zarr array with the specified codecs."""
        import zarr

        # Create a Zarr array in memory with the codecs
        zarr_array = zarr.create(
            shape=(1000, 1000),
            chunks=(100, 100),
            dtype="int32",
            store=None,
            zarr_format=zarr_format,
            # compressor=compressor,
            # filters=filters,
            codecs=codecs,
        )

        # Populate the Zarr array with data
        zarr_array[:] = np.arange(1000 * 1000).reshape(1000, 1000)
        return zarr_array

    def test_manifest_array_zarr_v3_default(self, create_manifestarray):
        """Test get_codecs with ManifestArray using default v3 codec."""
        manifest_array = create_manifestarray(codecs=None)
        actual_codecs = get_codecs(manifest_array)
        expected_codecs = tuple([BytesCodec(endian="little")])
        assert actual_codecs == expected_codecs

    def test_manifest_array_zarr_v3_with_codecs(self, create_manifestarray):
        """Test get_codecs with ManifestArray using multiple v3 codecs."""
        test_codecs = [DELTA_CODEC, ARRAYBYTES_CODEC, BLOSC_CODEC]
        manifest_array = create_manifestarray(codecs=test_codecs)
        actual_codecs = get_codecs(manifest_array)
        assert actual_codecs == tuple(
            [
                get_codec_class(codec["name"])(**codec["configuration"])
                for codec in test_codecs
            ]
        )

    def test_zarr_v3_default(self):
        """Test get_codecs with Zarr array using default v3 codec."""
        zarr_array = self.create_zarr_array()
        actual_codecs = get_codecs(zarr_array)
        assert isinstance(actual_codecs[0], BytesCodec)

    def test_zarr_v3_with_codecs(self):
        """Test get_codecs with Zarr array using multiple v3 codecs."""
        test_codecs = [DELTA_CODEC, ARRAYBYTES_CODEC, BLOSC_CODEC]
        zarr_array = self.create_zarr_array(codecs=test_codecs)
        actual_codecs = get_codecs(zarr_array)
        assert actual_codecs == tuple(
            [
                get_codec_class(codec["name"])(**codec["configuration"])
                for codec in test_codecs
            ]
        )

    def test_zarr_v2_error(self):
        """Test that using v2 format raises an error."""
        zarr_array = self.create_zarr_array(zarr_format=2)
        with pytest.raises(
            ValueError,
            match="Only zarr v3 format arrays are supported. Please convert your array to v3 format.",
        ):
            get_codecs(zarr_array)


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
                    get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC),
                ),  # type: ignore[arg-type]
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
                    get_codec_class("numcodecs.delta").from_dict(DELTA_CODEC),
                ),  # type: ignore[arg-type]
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
def test_convert_to_codec_pipeline_scenarios(input_codecs, expected_pipeline):
    """Test different scenarios for convert_to_codec_pipeline function."""
    dtype = np.dtype("<i4")
    if input_codecs is not None:
        input_codecs = list(input_codecs)

    result = convert_to_codec_pipeline(dtype=dtype, codecs=input_codecs)
    assert result == expected_pipeline
