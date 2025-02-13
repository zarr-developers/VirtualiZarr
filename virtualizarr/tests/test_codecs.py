import numpy as np
import pytest
from zarr.codecs import BytesCodec
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.registry import get_codec_class

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

    def test_manifest_array_zarr_v3_with_codecs(
        self, create_manifestarray, delta_codec, arraybytes_codec, blosc_codec
    ):
        """Test get_codecs with ManifestArray using multiple v3 codecs."""
        test_codecs = [
            delta_codec,
            arraybytes_codec,
            blosc_codec,
        ]
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

    def test_zarr_v3_with_codecs(self, delta_codec, arraybytes_codec, blosc_codec):
        """Test get_codecs with Zarr array using multiple v3 codecs."""
        test_codecs = [
            delta_codec,
            arraybytes_codec,
            blosc_codec,
        ]
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


def test_convert_to_codec_pipeline():
    expected_default_codecs = BatchedCodecPipeline(
        array_array_codecs=(),
        array_bytes_codec=BytesCodec(endian="little"),
        bytes_bytes_codecs=(),
        batch_size=1,
    )
    # Test with just dtype (default codec pipeline)
    dtype = np.dtype("<i4")
    int_codecs = convert_to_codec_pipeline(dtype=dtype)
    assert int_codecs == expected_default_codecs

    # Test with different dtype
    float_dtype = np.dtype("<f8")
    float_codecs = convert_to_codec_pipeline(dtype=float_dtype)
    assert float_codecs == expected_default_codecs

    # Test with empty codecs list
    empty_codecs = convert_to_codec_pipeline(dtype=dtype, codecs=[])
    assert empty_codecs == expected_default_codecs

    # Test with filters and compressor
    test_codecs = [
        {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}},
        {
            "name": "numcodecs.blosc",
            "configuration": {"cname": "zstd", "clevel": 5, "shuffle": 1},
        },
    ]

    codecs = convert_to_codec_pipeline(dtype=dtype, codecs=test_codecs)
    assert isinstance(codecs, BatchedCodecPipeline)

    # Verify codec types and order
    array_array_codecs = codecs.array_array_codecs
    assert array_array_codecs[0].codec_name == "numcodecs.delta"
    array_bytes_codec = codecs.array_bytes_codec
    assert isinstance(array_bytes_codec, BytesCodec)
    bytes_bytes_codecs = codecs.bytes_bytes_codecs
    assert bytes_bytes_codecs[0].codec_name == "numcodecs.blosc"
    config = bytes_bytes_codecs[0].codec_config
    assert config["cname"] == "zstd"
    assert config["clevel"] == 5
    assert config["shuffle"] == 1
