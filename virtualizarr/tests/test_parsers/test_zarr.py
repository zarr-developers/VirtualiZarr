import numpy as np
import pytest
import zarr
from obstore.store import LocalStore

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.parsers import ZarrParser

# from virtualizarr.parsers.zarr import get_chunk_mapping_prefix, get_metadata
from virtualizarr.registry import ObjectStoreRegistry

ZarrArrayType = zarr.AsyncArray | zarr.Array


@pytest.mark.parametrize(
    "zarr_store",
    [
        pytest.param(
            2,
            id="Zarr V2",
            # marks=pytest.mark.skip(reason="Zarr V2 not currently supported."),
        ),
        pytest.param(3, id="Zarr V3"),
    ],
    indirect=True,
)
class TestOpenVirtualDatasetZarr:
    def test_loadable_variables(self, zarr_store, loadable_variables=["time", "air"]):
        # check loadable variables
        store = LocalStore(prefix=zarr_store)
        registry = ObjectStoreRegistry({f"file://{zarr_store}": store})
        parser = ZarrParser()
        with open_virtual_dataset(
            url=zarr_store,
            registry=registry,
            parser=parser,
            loadable_variables=loadable_variables,
        ) as vds:
            assert isinstance(vds["time"].data, np.ndarray)
            assert isinstance(vds["air"].data, np.ndarray), type(vds["air"].data)

    def test_skip_variables(self, zarr_store, skip_variables=["air"]):
        store = LocalStore(prefix=zarr_store)
        registry = ObjectStoreRegistry({f"file://{zarr_store}": store})

        parser = ZarrParser(skip_variables=skip_variables)
        # check variable is skipped
        with open_virtual_dataset(
            url=zarr_store,
            registry=registry,
            parser=parser,
        ) as vds:
            assert len(vds.data_vars) == 0

    def test_manifest_indexing(self, zarr_store):
        store = LocalStore(prefix=zarr_store)
        registry = ObjectStoreRegistry({f"file://{zarr_store}": store})
        parser = ZarrParser()
        with open_virtual_dataset(
            url=zarr_store,
            registry=registry,
            parser=parser,
        ) as vds:
            assert "0.0.0" in vds["air"].data.manifest.dict().keys()

    def test_virtual_dataset_zarr_attrs(self, zarr_store):
        zg = zarr.open_group(zarr_store)
        store = LocalStore(prefix=zarr_store)
        registry = ObjectStoreRegistry({f"file://{zarr_store}": store})
        parser = ZarrParser()
        with open_virtual_dataset(
            url=zarr_store,
            registry=registry,
            parser=parser,
            loadable_variables=[],
        ) as vds:
            non_var_arrays = ["time", "lat", "lon"]

            # check dims and coords are present
            assert set(vds.coords) == set(non_var_arrays)
            assert set(vds.sizes) == set(non_var_arrays)
            # check vars match
            assert set(vds.keys()) == set(["air"])

            # check top level attrs
            assert zg.attrs.asdict() == vds.attrs

            arrays = [val for val in zg.keys()]

            # arrays are ManifestArrays
            for array in arrays:
                # check manifest array ArrayV3Metadata dtype
                assert isinstance(vds[array].data, ManifestArray)
                # compare manifest array ArrayV3Metadata
                expected = zg[array].metadata.to_dict()
                # Check attributes
                assert expected["attributes"] == vds[array].attrs

                # Check dimension names - handling V2 vs V3 difference
                zarr_format = zg[array].metadata.zarr_format
                if zarr_format == 2:
                    # V2 stores dimensions in attributes
                    expected_dims = expected.get("attributes", {}).get(
                        "_ARRAY_DIMENSIONS", None
                    )
                    if expected_dims:
                        assert expected_dims == list(vds[array].dims)
                    # If no _ARRAY_DIMENSIONS, VirtualiZarr generates dimension names
                else:  # V3
                    assert list(expected["dimension_names"]) == list(vds[array].dims)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_scalar_chunk_mapping(tmpdir, zarr_format):
    """Test that scalar arrays produce correct chunk mappings for both V2 and V3."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import get_strategy

    # Create a scalar zarr array
    filepath = f"{tmpdir}/scalar.zarr"
    scalar_array = zarr.create(
        shape=(), dtype="int8", store=filepath, zarr_format=zarr_format
    )
    scalar_array[()] = 42

    # Open it as an async array to use with the strategy
    async def get_chunk_map():
        zarr_array = await open_array(store=filepath, mode="r")
        strategy = get_strategy(zarr_array)
        return await strategy.get_chunk_mapping(zarr_array, filepath)

    chunk_map = asyncio.run(get_chunk_map())

    # V2 uses "0" for scalar, V3 uses "c"
    expected_key = "0" if zarr_format == 2 else "c"
    assert expected_key in chunk_map
    assert chunk_map[expected_key]["offset"] == 0
    assert chunk_map[expected_key]["length"] > 0


def test_join_url_empty_base():
    """Test join_url with empty base."""
    from virtualizarr.parsers.zarr import join_url

    result = join_url("", "some/key")
    assert result == "some/key"


def test_unsupported_zarr_format():
    """Test that unsupported zarr format raises NotImplementedError."""
    from unittest.mock import Mock

    from virtualizarr.parsers.zarr import get_strategy

    # Create a mock array with unsupported format
    mock_array = Mock()
    mock_array.metadata.zarr_format = 99  # Unsupported format

    with pytest.raises(NotImplementedError, match="Zarr format 99 is not supported"):
        get_strategy(mock_array)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_empty_array_chunk_mapping(tmpdir, zarr_format):
    """Test chunk mapping for arrays with no chunks written yet."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import get_strategy

    # Create an array but don't write any data
    filepath = f"{tmpdir}/empty.zarr"
    zarr.create(
        shape=(10, 10),
        chunks=(5, 5),
        dtype="int8",
        store=filepath,
        zarr_format=zarr_format,
    )

    async def get_chunk_map():
        zarr_array = await open_array(store=filepath, mode="r")
        strategy = get_strategy(zarr_array)
        return await strategy.get_chunk_mapping(zarr_array, filepath)

    chunk_map = asyncio.run(get_chunk_map())
    # Empty arrays should return empty chunk map
    assert chunk_map == {}


def test_v2_metadata_without_dimensions():
    """Test V2 metadata conversion when array has no _ARRAY_DIMENSIONS attribute."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import get_metadata

    # Create a V2 array without dimension attributes
    store = zarr.storage.MemoryStore()
    _ = zarr.create(
        shape=(5, 10), chunks=(5, 5), dtype="int32", store=store, zarr_format=2
    )
    # Explicitly don't set _ARRAY_DIMENSIONS

    async def get_meta():
        zarr_array = await open_array(store=store, mode="r")
        return get_metadata(zarr_array)

    metadata = asyncio.run(get_meta())
    # Should generate dimension names
    assert metadata.dimension_names is not None
    assert len(metadata.dimension_names) == 2


def test_v2_metadata_with_dimensions():
    """Test V2 metadata conversion when array has _ARRAY_DIMENSIONS attribute."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import get_metadata

    # Create a V2 array with dimension attributes
    store = zarr.storage.MemoryStore()
    array = zarr.create(
        shape=(5, 10), chunks=(5, 5), dtype="int32", store=store, zarr_format=2
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]

    async def get_meta():
        zarr_array = await open_array(store=store, mode="r")
        return get_metadata(zarr_array)

    metadata = asyncio.run(get_meta())
    # Should use the provided dimension names
    assert metadata.dimension_names == ("x", "y")


def test_v2_metadata_with_none_fill_value():
    """Test V2 metadata conversion when fill_value is None."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import get_metadata

    # Create a V2 array with None fill_value
    store = zarr.storage.MemoryStore()
    _ = zarr.create(
        shape=(5, 10),
        chunks=(5, 5),
        dtype="int32",
        store=store,
        zarr_format=2,
        fill_value=None,
    )

    async def get_meta():
        zarr_array = await open_array(store=store, mode="r")
        return get_metadata(zarr_array)

    metadata = asyncio.run(get_meta())
    # Should handle None fill_value gracefully
    assert metadata.fill_value is not None


def test_build_chunk_manifest_empty_with_shape():
    """Test build_chunk_manifest when chunk_map is empty but array has shape and chunks."""
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import build_chunk_manifest

    # Create an array but don't write data
    store = zarr.storage.MemoryStore()
    zarr.create(shape=(10, 10), chunks=(5, 5), dtype="int8", store=store, zarr_format=3)

    async def get_manifest():
        zarr_array = await open_array(store=store, mode="r")
        return await build_chunk_manifest(zarr_array, "test://path")

    manifest = asyncio.run(get_manifest())
    # Should create manifest with proper chunk grid shape even if empty
    assert manifest.shape_chunk_grid == (2, 2)  # 10/5 = 2 chunks per dimension


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_sparse_array_with_missing_chunks(tmpdir, zarr_format):
    """Test that arrays with some missing chunks (sparse arrays) are handled correctly.

    This test verifies that VirtualiZarr correctly handles the case where some chunks
    exist but others are missing. Zarr allows this for sparse data, and when chunks
    are missing, Zarr returns the fill_value for those regions. VirtualiZarr should
    preserve this sparsity in the manifest rather than generating entries for all
    possible chunks based on the chunk grid.
    """
    import asyncio

    from zarr.api.asynchronous import open_array

    from virtualizarr.parsers.zarr import build_chunk_manifest

    # Create a zarr array with a 3x3 chunk grid (9 possible chunks)
    filepath = f"{tmpdir}/sparse.zarr"
    arr = zarr.create(
        shape=(30, 30),
        chunks=(10, 10),
        dtype="float32",
        store=filepath,
        zarr_format=zarr_format,
        fill_value=np.nan,
    )

    # Only write data to some chunks, leaving others missing (sparse)
    # Write to chunks (0,0), (1,1), and (2,2) - a diagonal pattern
    arr[0:10, 0:10] = 1.0  # chunk 0.0
    arr[10:20, 10:20] = 2.0  # chunk 1.1
    arr[20:30, 20:30] = 3.0  # chunk 2.2
    # Chunks (0,1), (0,2), (1,0), (1,2), (2,0), (2,1) are intentionally left unwritten

    async def get_manifest():
        zarr_array = await open_array(store=filepath, mode="r")
        return await build_chunk_manifest(zarr_array, filepath)

    manifest = asyncio.run(get_manifest())

    # The manifest should only contain the 3 chunks we actually wrote
    assert len(manifest.dict()) == 3, f"Expected 3 chunks, got {len(manifest.dict())}"

    # Verify the expected chunks are present
    assert "0.0" in manifest.dict(), "Chunk 0.0 should be present"
    assert "1.1" in manifest.dict(), "Chunk 1.1 should be present"
    assert "2.2" in manifest.dict(), "Chunk 2.2 should be present"

    # Verify missing chunks are not in the manifest
    missing_chunks = ["0.1", "0.2", "1.0", "1.2", "2.0", "2.1"]
    for chunk_key in missing_chunks:
        assert chunk_key not in manifest.dict(), (
            f"Chunk {chunk_key} should not be present (it's missing/sparse)"
        )

    # The chunk grid shape should still reflect the full array dimensions
    assert manifest.shape_chunk_grid == (3, 3), "Chunk grid should be 3x3"
