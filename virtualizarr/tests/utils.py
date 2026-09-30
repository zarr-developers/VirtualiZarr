from __future__ import annotations

import os
import tempfile
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore, ObjectStore, from_url

from virtualizarr.parsers import HDFParser

# Find location of pytest temporary data in what should be a cross-platform way. This should be the same as what pytest actually does - see https://docs.pytest.org/en/stable/how-to/tmp_path.html#temporary-directory-location-and-retention
# The realpath call is there to resolve any symbolic links, such as from /var/ to /private/var/ on MacOS, as Icechunk needs the entire URL prefix without symlinks.
# Icechunk also needs the final / to create VirtualChunkContainers.
PYTEST_TMP_DIRECTORY_URL_PREFIX = f"file://{os.path.realpath(tempfile.gettempdir())}/"


@contextmanager
def open_hdf5(path: str | Path, mode: str = "r") -> Iterator:
    """Open an HDF5 file with h5py, yielding a weak proxy to the file.

    The proxy stops working once the block exits, so the caller's name does not
    keep the file alive afterwards. That matters when a test then raises (e.g.
    ``pytest.xfail``): the traceback keeps the test's frame alive in a reference
    cycle, and h5py objects left for the cyclic garbage collector can be freed on
    a thread that deadlocks against h5py's global lock (e.g. zarr's IO thread
    during kerchunk's ``SingleHdf5ToZarr.translate``). Child objects such as
    datasets are not proxied, so read what you need inside the block instead of
    binding them to names.
    """
    import h5py

    with h5py.File(path, mode) as f:
        yield weakref.proxy(f)


def obstore_local(url: str) -> ObjectStore:
    parsed = urlparse(url)
    path = Path(parsed.path)
    store = LocalStore(prefix=path.parent)
    return store


def obstore_s3(url: str, region: str) -> ObjectStore:
    parsed = urlparse(url)
    bucket = parsed.netloc
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"s3://{bucket}/{key_prefix}"
    store = from_url(url=base_path, region=region, skip_signature=True)
    return store


def obstore_http(url: str) -> ObjectStore:
    parsed = urlparse(url)
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"{parsed.scheme}://{parsed.netloc}/{key_prefix}"
    store = from_url(url=base_path)
    return store


def manifest_store_from_hdf_url(url, group: str | None = None):
    registry: ObjectStoreRegistry = ObjectStoreRegistry()
    registry.register(url, obstore_local(url=url))
    parser = HDFParser(group=group)
    return parser(url=url, registry=registry)
