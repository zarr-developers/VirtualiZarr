import numpy as np
import pytest
from zarr.codecs import BytesCodec
from zarr.registry import get_codec_class

from virtualizarr.codecs import get_codecs

arrayarray_codec = {"name": "numcodecs.delta", "configuration": {"dtype": "<i8"}}
arraybytes_codec = {"name": "bytes", "configuration": {"endian": "little"}}
bytesbytes_codec = {
    "name": "blosc",
    "configuration": {
        "cname": "zstd",
        "clevel": 5,
        "shuffle": "shuffle",
        "typesize": 4,
    },
}

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
        test_codecs = [
            arrayarray_codec,
            arraybytes_codec,
            bytesbytes_codec,
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

    def test_zarr_v3_with_codecs(self):
        """Test get_codecs with Zarr array using multiple v3 codecs."""
        test_codecs = [
            arrayarray_codec,
            arraybytes_codec,
            bytesbytes_codec,
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
