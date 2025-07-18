import pytest
from obstore.store import MemoryStore

from virtualizarr.registry import ObjectStoreRegistry


def test_registry():
    registry = ObjectStoreRegistry()
    memstore = MemoryStore()
    registry.register("s3://bucket1", memstore)
    url = "s3://bucket1/path/to/object"
    ret, path = registry.resolve(url)
    assert path == "path/to/object"
    assert ret is memstore


def test_register_raises():
    registry = ObjectStoreRegistry()
    with pytest.raises(
        ValueError,
        match=r"Urls are expected to contain a scheme \(e\.g\., `file://` or `s3://`\), received .* which parsed to ParseResult\(scheme='.*', netloc='.*', path='.*', params='.*', query='.*', fragment='.*'\)",
    ):
        url = "bucket1/path/to/object"
        ret, path = registry.register(url, MemoryStore())


def test_resolve_raises():
    registry = ObjectStoreRegistry()
    with pytest.raises(
        ValueError,
        match="Could not find an ObjectStore matching the url `s3://bucket1/path/to/object`",
    ):
        url = "s3://bucket1/path/to/object"
        ret, path = registry.resolve(url)
