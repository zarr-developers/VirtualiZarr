from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from numcodecs import Blosc, Delta
from packaging.version import parse

from virtualizarr import ChunkManifest, ManifestArray
from virtualizarr.codecs import get_codecs
from virtualizarr.zarr import Codec


def requires_version(package_name, required_version):
    """
    Decorator to skip a test if the specified package is not installed or does not meet the required version.

    Args:
        package_name (str): Name of the package to check.
        required_version (str): Minimum required version of the package.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Attempt to import the package
            package = pytest.importorskip(package_name)
            # Check if the version meets the requirement
            if parse(package.__version__) < parse(required_version):
                pytest.skip(
                    f"'{package_name}' version >= {required_version} is required."
                )
            # Proceed with the test
            return func(*args, **kwargs)

        return wrapper

    return decorator


class TestCodecs:
    def create_manifest_array(self, compressor=None, filters=None, zarr_format=2):
        return ManifestArray(
            chunkmanifest=ChunkManifest(
                entries={"0.0": dict(path="/test.nc", offset=6144, length=48)}
            ),
            zarray=dict(
                shape=(2, 3),
                dtype=np.dtype("<i8"),
                chunks=(2, 3),
                compressor=compressor,
                filters=filters,
                fill_value=0,
                order="C",
                zarr_format=zarr_format,
            ),
        )

    def test_manifest_array_codecs_v2(self):
        """Test that get_codecs works for ManifestArray with Zarr v2 metadata."""
        compressor = {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1}
        filters = [{"id": "delta", "dtype": "<i8"}]
        manifest_array = self.create_manifest_array(
            zarr_format=2, compressor=compressor, filters=filters
        )

        # Get codecs and verify
        actual_codecs = get_codecs(manifest_array)
        expected_codecs = Codec(
            compressor={"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1},
            filters=[{"id": "delta", "dtype": "<i8"}],
        )
        assert actual_codecs == expected_codecs

    @requires_version("zarr", "3.0.0b")
    def test_manifest_array_codecs_v3(self):
        """Test that get_codecs works for ManifestArray with Zarr v2 metadata."""
        from zarr.codecs import BytesCodec

        manifest_array = self.create_manifest_array(zarr_format=3)

        # Get codecs and verify
        actual_codecs = get_codecs(manifest_array)
        expected_codecs = tuple([BytesCodec(endian="little")])
        assert actual_codecs == expected_codecs

    def create_zarr_array(
        self,
        compressor=None,
        filters=None,
        codecs=None,
        zarr_format=2,
    ):
        shared_kwargs = {
            "shape": (1000, 1000),
            "chunks": (100, 100),
            "dtype": "int32",
            "store": None,
            "zarr_format": zarr_format,
        }
        if zarr_format == 2:
            shared_kwargs["compressor"] = compressor
            shared_kwargs["filters"] = filters
        elif zarr_format == 3:
            shared_kwargs["codecs"] = codecs
        # Create a Zarr array in memory with the codecs
        zarr_array = zarr.create(**shared_kwargs)

        # Populate the Zarr array with data
        zarr_array[:] = np.arange(1000 * 1000).reshape(1000, 1000)

        return zarr_array

    def test_zarr_v2_array_codecs(self):
        # Define your codecs (compressor and filters)
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        filters = [Delta(dtype="int32")]

        zarr_array_v2 = self.create_zarr_array(compressor=compressor, filters=filters)
        # Test codecs
        actual_codecs = get_codecs(zarr_array_v2)
        expected_codecs = Codec(
            compressor=Blosc(
                cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE, blocksize=0
            ),
            filters=[
                Delta(dtype="<i4"),
            ],
        )
        assert actual_codecs == expected_codecs

    @requires_version("zarr", "3.0.0b")
    def test_zarr_v3_array_codecs(self):
        from zarr.codecs import BytesCodec

        zarr_array_v3 = self.create_zarr_array(
            codecs=[BytesCodec(endian="little")], zarr_format=3
        )

        # Test codecs
        actual_codecs = get_codecs(zarr_array_v3)
        expected_codecs = tuple([BytesCodec(endian="little")])
        assert actual_codecs == expected_codecs

    def test_unsupported_zarr_python(self):
        zarr_array = self.create_zarr_array()
        unsupported_zarr_version = "2.18.3"
        with patch("zarr.__version__", unsupported_zarr_version):
            try:
                get_codecs(zarr_array)
            except RuntimeError as e:
                assert (
                    str(e)
                    == f"zarr-python v3 or higher is required, but version {unsupported_zarr_version} is installed."
                )
