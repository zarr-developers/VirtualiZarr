import asyncio
from typing import cast

import numpy as np
import obstore
import pytest
import xarray as xr
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from obstore.store import MemoryStore as ObsMemoryStore
from packaging import version
from zarr.api.asynchronous import open_array
from zarr.storage import ObjectStore

from virtualizarr import open_virtual_dataset
from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.utils import ChunkKeySeparator
from virtualizarr.parsers import ZarrParser
from virtualizarr.parsers.zarr import (
    ZarrFormat,
    _run_async,
    build_chunk_manifest,
    join_url,
    metadata_as_v3,
)
from virtualizarr.tests import requires_arro3, requires_minio

pytestmark = requires_arro3

HAS_V2_MIGRATION = version.parse(zarr.__version__) >= version.parse("3.1.3")

requires_v2_migration = pytest.mark.skipif(
    not HAS_V2_MIGRATION,
    reason="V2→V3 metadata migration requires zarr>=3.1.3",
)


async def _build_manifest(zarr_store: ObjectStore, store_base_uri: str):
    """Helper to open an array from a zarr store and build its chunk manifest."""
    zarr_array = await open_array(store=zarr_store, mode="r")
    fmt = ZarrFormat(zarr_array.metadata.zarr_format)
    sep: ChunkKeySeparator = (
        zarr_array.metadata.chunk_key_encoding.separator
        if fmt == ZarrFormat.V3
        else "."
    )
    return await build_chunk_manifest(
        obs_store=cast(ObjectStore, zarr_array.store).store,
        array_path=zarr_array.path,
        store_base_uri=store_base_uri,
        metadata=metadata_as_v3(zarr_array.metadata),
        on_disk_zarr_format=fmt,
        on_disk_separator=sep,
    )


def zarr_versions(param_name="zarr_format", indirect=False):
    """
    Reusable parametrize decorator for Zarr V2 and V3 versions.

    Args:
        param_name: Name of the parameter ('zarr_format' or 'zarr_store')
        indirect: Whether to use indirect parametrization (True for fixtures)
    """
    return pytest.mark.parametrize(
        param_name,
        [
            pytest.param(2, id="Zarr V2", marks=requires_v2_migration),
            pytest.param(3, id="Zarr V3"),
        ],
        indirect=indirect,
    )


@zarr_versions(param_name="zarr_store", indirect=True)
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

                # Check attributes - V2 to V3 conversion removes _ARRAY_DIMENSIONS
                expected_attrs = expected["attributes"].copy()
                if "_ARRAY_DIMENSIONS" in expected_attrs:
                    # V2 stores dimensions in attributes, VirtualiZarr converts to V3 dimension_names
                    expected_dims = expected_attrs["_ARRAY_DIMENSIONS"]
                    del expected_attrs["_ARRAY_DIMENSIONS"]
                    assert expected_dims == list(vds[array].dims)
                else:  # V3
                    assert list(expected["dimension_names"]) == list(vds[array].dims)


@zarr_versions()
def test_scalar_chunk_mapping(tmpdir, zarr_format):
    """Test that scalar arrays produce correct chunk mappings for both V2 and V3."""

    # Create a scalar zarr array
    filepath = f"{tmpdir}/scalar.zarr"
    scalar_array = zarr.create(
        shape=(), dtype="int8", store=filepath, zarr_format=zarr_format
    )
    scalar_array[()] = 42

    zarr_store = ObjectStore(store=LocalStore(prefix=filepath))
    manifest = asyncio.run(_build_manifest(zarr_store, filepath))

    # scalar arrays have a single chunk with empty coordinate key
    chunk_dict = manifest.dict()
    assert "" in chunk_dict
    assert chunk_dict[""]["offset"] == 0
    assert chunk_dict[""]["length"] > 0


def test_join_url_empty_base():
    """Test join_url with empty base."""

    result = join_url("", "some/key")
    assert result == "some/key"


def test_unsupported_zarr_format():
    """Test that unsupported zarr format raises ValueError."""
    with pytest.raises(ValueError):
        ZarrFormat(99)


@zarr_versions()
def test_empty_array_chunk_mapping(tmpdir, zarr_format):
    """Test chunk mapping for arrays with no chunks written yet."""

    filepath = f"{tmpdir}/empty.zarr"
    zarr.create(
        shape=(10, 10),
        chunks=(5, 5),
        dtype="int8",
        store=filepath,
        zarr_format=zarr_format,
    )

    zarr_store = ObjectStore(store=LocalStore(prefix=filepath))
    manifest = asyncio.run(_build_manifest(zarr_store, filepath))
    assert manifest.dict() == {}


@requires_v2_migration
def test_v2_metadata_without_dimensions():
    """Test V2 metadata conversion when array has no _ARRAY_DIMENSIONS attribute."""
    store = zarr.storage.MemoryStore()
    zarr.create(shape=(5, 10), chunks=(5, 5), dtype="int32", store=store, zarr_format=2)

    metadata = metadata_as_v3(zarr.open(store, mode="r").metadata)
    assert metadata.dimension_names is None


@pytest.mark.skipif(HAS_V2_MIGRATION, reason="Test only relevant for zarr<3.1.3")
def test_v2_metadata_raises_import_error_on_old_zarr():
    """Test that V2 metadata conversion raises ImportError with zarr<3.1.3."""
    store = zarr.storage.MemoryStore()
    zarr.create(shape=(5, 10), chunks=(5, 5), dtype="int32", store=store, zarr_format=2)

    with pytest.raises(
        ImportError,
        match=r"Zarr-Python>=3\.1\.3 is required for parsing Zarr V2 into Zarr V3.*Found Zarr version",
    ):
        metadata_as_v3(zarr.open(store, mode="r").metadata)


@requires_v2_migration
def test_v2_metadata_with_dimensions():
    """Test V2 metadata conversion when array has _ARRAY_DIMENSIONS attribute."""
    store = zarr.storage.MemoryStore()
    array = zarr.create(
        shape=(5, 10), chunks=(5, 5), dtype="int32", store=store, zarr_format=2
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]

    metadata = metadata_as_v3(zarr.open(store, mode="r").metadata)
    assert metadata.dimension_names == ("x", "y")


def test_v3_metadata_separator_normalized():
    """Test that metadata_as_v3 normalizes V3 chunk_key_encoding separator to '.'."""
    store = zarr.storage.MemoryStore()
    zarr.create(
        shape=(5, 10),
        chunks=(5, 5),
        dtype="int32",
        store=store,
        zarr_format=3,
        chunk_key_encoding={"name": "default", "separator": "/"},
    )

    metadata = metadata_as_v3(zarr.open(store, mode="r").metadata)
    assert metadata.chunk_key_encoding.separator == "."


@requires_v2_migration
@pytest.mark.parametrize(
    "dtype",
    [
        "int32",
        "uint8",
        "float64",
        "bool",
        "U10",
        "datetime64[s]",
        "timedelta64[s]",
        "S10",
        "V10",
    ],
)
def test_v2_metadata_with_none_fill_value(dtype):
    """Test V2 metadata conversion when fill_value is None."""
    store = zarr.storage.MemoryStore()
    zarr.create(
        shape=(5, 10),
        chunks=(5, 5),
        dtype=dtype,
        store=store,
        zarr_format=2,
        fill_value=None,
    )

    metadata = metadata_as_v3(zarr.open(store, mode="r").metadata)
    assert metadata.fill_value is not None


def test_build_chunk_manifest_empty_with_shape():
    """Test build_chunk_manifest when chunk_map is empty but array has shape and chunks."""
    zarr_store = ObjectStore(store=ObsMemoryStore())
    zarr.create(
        shape=(10, 10), chunks=(5, 5), dtype="int8", store=zarr_store, zarr_format=3
    )

    manifest = asyncio.run(_build_manifest(zarr_store, "test://path"))
    assert manifest.shape_chunk_grid == (2, 2)


@zarr_versions()
def test_sparse_array_with_missing_chunks(tmpdir, zarr_format):
    """Test that arrays with some missing chunks (sparse arrays) are handled correctly."""
    filepath = f"{tmpdir}/sparse.zarr"
    arr = zarr.create(
        shape=(30, 30),
        chunks=(10, 10),
        dtype="float32",
        store=filepath,
        zarr_format=zarr_format,
        fill_value=np.nan,
    )

    arr[0:10, 0:10] = 1.0  # chunk 0.0
    arr[10:20, 10:20] = 2.0  # chunk 1.1
    arr[20:30, 20:30] = 3.0  # chunk 2.2

    zarr_store = ObjectStore(store=LocalStore(prefix=filepath))
    manifest = asyncio.run(_build_manifest(zarr_store, filepath))

    assert len(manifest.dict()) == 3
    assert "0.0" in manifest.dict()
    assert "1.1" in manifest.dict()
    assert "2.2" in manifest.dict()

    missing_chunks = ["0.1", "0.2", "1.0", "1.2", "2.0", "2.1"]
    for chunk_key in missing_chunks:
        assert chunk_key not in manifest.dict()

    assert manifest.shape_chunk_grid == (3, 3)


@zarr_versions()
def test_parser_roundtrip_matches_xarray(tmpdir, zarr_format):
    """Roundtrip a small dataset through the ZarrParser and compare with xarray."""

    # Create a small Dataset with chunking
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(36).reshape(6, 6).astype("float32"))},
        coords={"x": np.arange(6), "y": np.arange(6)},
    )

    filepath = f"{tmpdir}/roundtrip.zarr"
    # Ensure multiple chunks to exercise manifest generation
    ds.to_zarr(
        filepath,
        encoding={"data": {"chunks": (2, 2)}},
        consolidated=False,
        zarr_format=zarr_format,
    )

    # Build a registry and generate a ManifestStore from the parser
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()
    manifeststore = parser(url=filepath, registry=registry)

    # Open the original zarr and the manifest-backed store and compare
    with xr.open_dataset(
        filepath, engine="zarr", consolidated=False, zarr_format=zarr_format
    ) as expected:
        with xr.open_dataset(
            manifeststore, engine="zarr", consolidated=False, zarr_format=3
        ) as actual:
            xr.testing.assert_identical(actual, expected)


@zarr_versions()
def test_parser_scalar_roundtrip_matches_xarray(tmpdir, zarr_format):
    """Roundtrip a small dataset through the ZarrParser and compare with xarray."""

    # Create a small Dataset with a scalar
    ds = xr.Dataset(
        {"data": 42.0},
    )

    filepath = f"{tmpdir}/roundtrip.zarr"
    # Ensure multiple chunks to exercise manifest generation
    ds.to_zarr(
        filepath,
        consolidated=False,
        zarr_format=zarr_format,
    )

    # Build a registry and generate a ManifestStore from the parser
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()
    manifeststore = parser(url=filepath, registry=registry)

    # Open the original zarr and the manifest-backed store and compare
    with xr.open_dataset(
        filepath, engine="zarr", consolidated=False, zarr_format=zarr_format
    ) as expected:
        with xr.open_dataset(
            manifeststore, engine="zarr", consolidated=False, zarr_format=3
        ) as actual:
            xr.testing.assert_identical(actual, expected)


def test_run_async_without_running_loop():
    """Test _run_async works normally when no event loop is running."""

    async def coro():
        return 42

    assert _run_async(coro()) == 42


def test_run_async_with_running_loop():
    """Test _run_async works inside a running event loop (e.g. Jupyter notebooks).

    This simulates the notebook environment where asyncio.run() would raise
    RuntimeError because an event loop is already running.
    """

    async def coro():
        return 42

    async def outer():
        # We're inside a running loop here, so asyncio.run() would fail.
        return _run_async(coro())

    result = asyncio.run(outer())
    assert result == 42


@zarr_versions()
def test_zarr_parser_works_inside_running_event_loop(tmpdir, zarr_format):
    """Test that ZarrParser.__call__ works inside a running event loop (notebook scenario)."""

    ds = xr.Dataset(
        {"data": (("x",), np.arange(10, dtype="float32"))},
    )
    filepath = f"{tmpdir}/loop_test.zarr"
    ds.to_zarr(filepath, consolidated=False, zarr_format=zarr_format)

    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()

    async def run_parser_in_loop():
        return parser(url=filepath, registry=registry)

    manifest_store = asyncio.run(run_parser_in_loop())
    with xr.open_dataset(
        manifest_store, engine="zarr", consolidated=False, zarr_format=3
    ) as actual:
        with xr.open_dataset(
            filepath, engine="zarr", consolidated=False, zarr_format=zarr_format
        ) as expected:
            xr.testing.assert_identical(actual, expected)


@zarr_versions()
def test_parser_with_nested_store_path(tmpdir, zarr_format):
    """Regression test for https://github.com/zarr-developers/VirtualiZarr/issues/912."""
    parent_dir = f"{tmpdir}/bucket_root"
    # Use path names whose characters don't overlap with the variable name "temp",
    # to avoid triggering a separate zarr list_dir bug (zarr-developers/zarr-python#3657)
    filepath = f"{parent_dir}/foo/bar.zarr"

    ds = xr.Dataset(
        {"temp": (("x", "y"), np.arange(12, dtype="float32").reshape(3, 4))},
    )
    ds.to_zarr(filepath, consolidated=False, zarr_format=zarr_format)

    store = LocalStore(prefix=parent_dir)
    registry = ObjectStoreRegistry({f"file://{parent_dir}": store})
    parser = ZarrParser()

    manifeststore = parser(url=filepath, registry=registry)

    with xr.open_dataset(
        filepath, engine="zarr", consolidated=False, zarr_format=zarr_format
    ) as expected:
        with xr.open_dataset(
            manifeststore, engine="zarr", consolidated=False, zarr_format=3
        ) as actual:
            xr.testing.assert_identical(actual, expected)


def test_sharded_array_raises_error(tmpdir):
    """Test that attempting to virtualize a sharded Zarr V3 array raises NotImplementedError."""
    filepath = f"{tmpdir}/test_sharded.zarr"

    # Create a Zarr V3 group with a sharded array
    root = zarr.open_group(store=filepath, mode="w", zarr_format=3)
    root.create_array(
        name="data",
        shape=(100, 100),
        chunks=(10, 10),
        shards=(50, 50),  # This adds sharding
        dtype="float32",
    )

    # Attempt to open with VirtualiZarr should raise NotImplementedError
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()

    with pytest.raises(
        NotImplementedError,
        match="Zarr V3 arrays with sharding are not yet supported",
    ):
        parser(url=filepath, registry=registry)


@requires_minio
@pytest.mark.xfail(
    reason="ZarrParser does not yet support buckets without list permissions"
)
def test_zarr_parser_nolist_bucket(minio_nolist_bucket):
    """Test that ZarrParser works with a bucket that does not allow list operations."""
    bucket = minio_nolist_bucket["bucket"]
    endpoint = minio_nolist_bucket["endpoint"]
    username = minio_nolist_bucket["username"]
    password = minio_nolist_bucket["password"]

    # Write a Zarr V3 store directly to the bucket using admin credentials
    admin_store = obstore.store.S3Store(
        bucket,
        endpoint_url=endpoint,
        access_key_id=username,
        secret_access_key=password,
        virtual_hosted_style_request=False,
        client_options={"allow_http": True},
    )
    zarr_store = zarr.storage.ObjectStore(store=admin_store)
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12, dtype="float32").reshape(3, 4))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    ds.to_zarr(zarr_store, consolidated=False, zarr_format=3)

    # Create an anonymous S3 store (subject to bucket policy which denies list)
    anon_store = obstore.store.S3Store(
        bucket,
        endpoint_url=endpoint,
        skip_signature=True,
        virtual_hosted_style_request=False,
        client_options={"allow_http": True},
    )

    url = f"s3://{bucket}"
    registry = ObjectStoreRegistry({url: anon_store})
    parser = ZarrParser()
    manifeststore = parser(url=url, registry=registry)

    with xr.open_dataset(
        manifeststore, engine="zarr", consolidated=False, zarr_format=3
    ) as actual:
        xr.testing.assert_identical(actual, ds)
