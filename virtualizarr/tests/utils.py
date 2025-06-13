from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from obstore.store import LocalStore, ObjectStore, from_url


def obstore_local(file_url: str) -> ObjectStore:
    path = Path(file_url)
    store = LocalStore(prefix=path.parent)
    return store


def obstore_s3(file_url: str, region: str) -> ObjectStore:
    parsed = urlparse(file_url)
    bucket = parsed.netloc
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"s3://{bucket}/{key_prefix}"
    store = from_url(url=base_path, region=region, skip_signature=True)
    return store


def obstore_http(file_url: str) -> ObjectStore:
    parsed = urlparse(file_url)
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"{parsed.scheme}://{parsed.netloc}/{key_prefix}"
    store = from_url(url=base_path)
    return store
