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
from virtualizarr.parsers.zarr.common import (
    ZarrFormat,
    _run_async,
    join_url,
    metadata_as_v3,
)
from virtualizarr.parsers.zarr.zarr import build_chunk_manifest
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


@zarr_versions()
def test_uninitialized_scalar_chunk_mapping(tmpdir, zarr_format):
    """Test chunk mapping for a scalar array whose chunk was never written.

    This is common for CF grid-mapping / CRS variables, which are scalar arrays
    that carry only attributes and hold no data, so their single chunk is
    uninitialized. The parser must produce an empty (all-fill) manifest rather
    than raising.
    """

    filepath = f"{tmpdir}/uninitialized_scalar.zarr"
    zarr.create(shape=(), dtype="int8", store=filepath, zarr_format=zarr_format)

    zarr_store = ObjectStore(store=LocalStore(prefix=filepath))
    manifest = asyncio.run(_build_manifest(zarr_store, filepath))
    assert manifest.dict() == {}
    assert manifest.shape_chunk_grid == ()


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


@requires_v2_migration
def test_v2_metadata_with_scalar_dimensions():
    """Test V2 metadata conversion for scalar array with _ARRAY_DIMENSIONS=[]."""
    store = zarr.storage.MemoryStore()
    array = zarr.create(shape=(), chunks=(), dtype="int64", store=store, zarr_format=2)
    array.attrs["_ARRAY_DIMENSIONS"] = []

    metadata = metadata_as_v3(zarr.open(store, mode="r").metadata)
    assert metadata.dimension_names == ()


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


def test_build_chunk_manifest_skips_normalized_directory_marker():
    """Zero-byte directory markers remain non-chunks after path normalization."""
    from arro3.core import Array, RecordBatch

    class NormalizedDirectoryMarkerStore:
        async def list_async(self, *, prefix, return_arrow):
            assert prefix == "x/"
            assert return_arrow is True
            paths = Array.from_numpy(np.array(["x/0", "x"], dtype="U3"))
            sizes = Array.from_numpy(np.array([32, 0], dtype=np.uint64))
            yield RecordBatch.from_arrays([paths, sizes], names=["path", "size"])

    metadata_store = ObjectStore(store=ObsMemoryStore())
    zarr_array = zarr.create(
        shape=(4,),
        chunks=(4,),
        dtype="int64",
        store=metadata_store,
        zarr_format=2,
    )
    metadata = metadata_as_v3(zarr_array.metadata)

    manifest = asyncio.run(
        build_chunk_manifest(
            obs_store=NormalizedDirectoryMarkerStore(),
            array_path="x",
            store_base_uri="memory://bucket/store.zarr",
            metadata=metadata,
            on_disk_zarr_format=ZarrFormat.V2,
            on_disk_separator=".",
        )
    )

    chunk_dict = manifest.dict()
    assert set(chunk_dict) == {"0"}
    assert next(iter(chunk_dict.values()))["length"] == 32


@pytest.mark.parametrize(
    "zarr_format,separator,listed_keys,expected_key",
    [
        # marker for a chunk subdirectory, which only exists when chunk keys are
        # themselves nested (V2 dimension_separator="/", or any V3 array)
        pytest.param(
            ZarrFormat.V2,
            "/",
            {"x/0/0": 4, "x/0": 0, "x": 0},
            "0.0",
            id="v2-nested-marker",
        ),
        pytest.param(
            ZarrFormat.V3,
            "/",
            {"x/c/0/0": 4, "x/c/0": 0, "x/c": 0},
            "0.0",
            id="v3-nested-marker",
        ),
        # an object keyed at the chunks prefix itself that isn't zero-byte, so
        # isn't recognisable as a directory marker by its size
        pytest.param(
            ZarrFormat.V2,
            "/",
            {"x/0/0": 4, "x": 7},
            "0.0",
            id="v2-nonempty-object-at-prefix",
        ),
    ],
)
def test_build_chunk_manifest_skips_nested_directory_markers(
    zarr_format, separator, listed_keys, expected_key
):
    """Only keys shaped like a genuine chunk key (one coordinate component per
    dimension, nested under the chunks prefix) are treated as chunks.

    obstore strips the trailing slash from a listed directory marker's key, so a
    marker for a *nested* chunk subdirectory (e.g. ``x/0/`` alongside the chunk
    ``x/0/0``) arrives looking like a chunk key one component short. Regression
    test for the nested case left open by #1054.
    """
    # arro3 is an optional dependency, so it can't be imported at module scope
    from arro3.core import Array, RecordBatch

    class FabricatedListingStore:
        async def list_async(self, *, prefix, return_arrow):
            paths = Array.from_numpy(np.array(list(listed_keys), dtype="U16"))
            sizes = Array.from_numpy(
                np.array(list(listed_keys.values()), dtype=np.uint64)
            )
            yield RecordBatch.from_arrays([paths, sizes], names=["path", "size"])

    metadata_store = ObjectStore(store=ObsMemoryStore())
    zarr_array = zarr.create(
        shape=(10, 10),
        chunks=(5, 5),
        dtype="int8",
        store=metadata_store,
        zarr_format=zarr_format.value,
    )
    metadata = metadata_as_v3(zarr_array.metadata)

    manifest = asyncio.run(
        build_chunk_manifest(
            obs_store=FabricatedListingStore(),
            array_path="x",
            store_base_uri="memory://bucket/store.zarr",
            metadata=metadata,
            on_disk_zarr_format=zarr_format,
            on_disk_separator=separator,
        )
    )

    chunk_dict = manifest.dict()
    assert set(chunk_dict) == {expected_key}
    assert chunk_dict[expected_key]["length"] == 4


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
def test_parser_recurses_into_subgroups(tmpdir, zarr_format):
    """ZarrParser should virtualize arrays nested in subgroups, not just the root group.

    Regression test: previously construct_manifest_group only collected root-level
    arrays and dropped all subgroups silently.
    """
    filepath = f"{tmpdir}/hierarchical.zarr"

    # Array names deliberately share the "var" substring across levels: zarr<3.1.6
    # mis-strips keys when listing a nested group, silently dropping such arrays
    # (zarr-developers/zarr-python#3657). The `zarr>=3.1.6` floor guards against it;
    # these names would regress on an older zarr.
    dt = xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"root_var": (("x",), np.arange(4, dtype="float32"))}),
            "/group_a": xr.Dataset({"a_var": (("y",), np.arange(6, dtype="float32"))}),
            "/group_a/nested": xr.Dataset(
                {"deep_var": (("z",), np.arange(8, dtype="float32"))}
            ),
        }
    )
    dt.to_zarr(filepath, consolidated=False, zarr_format=zarr_format)

    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()
    manifeststore = parser(url=filepath, registry=registry)

    # nested arrays surface through the recursion, with the full hierarchy intact
    manifest_group = manifeststore._group
    assert "root_var" in manifest_group.arrays
    assert "group_a" in manifest_group.groups
    assert "a_var" in manifest_group.groups["group_a"].arrays
    assert "nested" in manifest_group.groups["group_a"].groups
    assert "deep_var" in manifest_group.groups["group_a"].groups["nested"].arrays

    # and the hierarchy round-trips end-to-end through a datatree
    with xr.open_datatree(
        filepath, engine="zarr", consolidated=False, zarr_format=zarr_format
    ) as expected:
        with xr.open_datatree(
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


def test_sharded_array_roundtrip(tmpdir):
    """Test that a sharded Zarr V3 array can be virtualized and read back correctly."""
    filepath = f"{tmpdir}/test_sharded.zarr"

    # Create a small sharded dataset via xarray
    ds = xr.Dataset(
        {"data": (("x", "y"), np.arange(12 * 12, dtype="float32").reshape(12, 12))},
    )
    ds.to_zarr(
        filepath,
        encoding={"data": {"chunks": (3, 3), "shards": (6, 6)}},
        consolidated=False,
        zarr_format=3,
    )

    # Parse with VirtualiZarr
    store = LocalStore(prefix=filepath)
    registry = ObjectStoreRegistry({f"file://{filepath}": store})
    parser = ZarrParser()
    manifeststore = parser(url=filepath, registry=registry)

    # Read back via ManifestStore and compare to original
    with xr.open_dataset(
        filepath, engine="zarr", consolidated=False, zarr_format=3
    ) as expected:
        with xr.open_dataset(
            manifeststore, engine="zarr", consolidated=False, zarr_format=3
        ) as actual:
            xr.testing.assert_identical(actual, expected)


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


@requires_v2_migration
def test_v2_slash_dimension_separator(tmp_path, local_registry):
    """Zarr V2 stores may use dimension_separator="/", giving chunk keys like "data/0/0"."""
    store_path = tmp_path / "slash_sep.zarr"
    group = zarr.create_group(str(store_path), zarr_format=2)
    arr = group.create_array(
        "data",
        shape=(4, 6),
        chunks=(2, 3),
        dtype="float64",
        chunk_key_encoding={"name": "v2", "separator": "/"},
        attributes={"_ARRAY_DIMENSIONS": ["x", "y"]},
    )
    expected = np.arange(24, dtype="float64").reshape(4, 6)
    arr[:] = expected

    with open_virtual_dataset(
        url=f"file://{store_path}",
        registry=local_registry,
        parser=ZarrParser(),
        loadable_variables=[],
    ) as vds:
        manifest = vds["data"].data.manifest.dict()
        assert set(manifest.keys()) == {"0.0", "0.1", "1.0", "1.1"}

    with open_virtual_dataset(
        url=f"file://{store_path}",
        registry=local_registry,
        parser=ZarrParser(),
        loadable_variables=["data"],
    ) as vds:
        np.testing.assert_array_equal(vds["data"].values, expected)


@pytest.mark.parametrize("group_path", ["", "nested"], ids=["root", "in-group"])
def test_v3_dot_chunk_key_separator(tmp_path, local_registry, group_path):
    """Zarr V3 arrays may use a "." chunk key separator, giving chunk keys like
    "data/c.0.0" rather than "data/c/0/0". Regression test for #1069."""
    store_path = tmp_path / "dot_sep.zarr"
    root = zarr.create_group(str(store_path), zarr_format=3)
    group = root.create_group(group_path) if group_path else root
    arr = group.create_array(
        "data",
        shape=(4, 6),
        chunks=(2, 3),
        dtype="float64",
        chunk_key_encoding={"name": "default", "separator": "."},
        dimension_names=["x", "y"],
    )
    expected = np.arange(24, dtype="float64").reshape(4, 6)
    arr[:] = expected

    with open_virtual_dataset(
        url=f"file://{store_path}",
        registry=local_registry,
        parser=ZarrParser(group=group_path or None),
        loadable_variables=[],
    ) as vds:
        manifest = vds["data"].data.manifest.dict()
        assert set(manifest.keys()) == {"0.0", "0.1", "1.0", "1.1"}
        # the manifest must point at the real (dot-separated) chunk keys
        assert manifest["0.0"]["path"].endswith(f"{group_path}/data/c.0.0".lstrip("/"))

    with open_virtual_dataset(
        url=f"file://{store_path}",
        registry=local_registry,
        parser=ZarrParser(group=group_path or None),
        loadable_variables=["data"],
    ) as vds:
        np.testing.assert_array_equal(vds["data"].values, expected)
