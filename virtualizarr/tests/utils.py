from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from obstore import open_reader
from obstore.store import LocalStore, ObjectStore, from_url

if TYPE_CHECKING:
    from obstore import ReadableFile

@dataclasses.dataclass
class BackendArguments:
    filepath: str
    file: ReadableFile 
    object_reader: ObjectStore

def obstore_local(filepath: str) -> BackendArguments:
    path = Path(filepath)
    store = LocalStore(prefix=path.parent)
    file = open_reader(store=store, path=path.name)
    return BackendArguments(
        filepath=filepath,
        file=file,
        object_reader=store,
    )

def obstore_s3(filepath: str, region: str) -> BackendArguments:
    parsed = urlparse(filepath)
    bucket = parsed.netloc
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"s3://{bucket}/{key_prefix}"
    filename = os.path.basename(parsed.path)
    store = from_url(url=base_path, region=region, skip_signature=True)
    file = open_reader(store=store, path=filename)
    return BackendArguments(
        filepath=filepath,
        file=file,
        object_reader=store,
    )

def obstore_http(filepath: str) -> BackendArguments:
    parsed = urlparse(filepath)
    key_prefix = os.path.dirname(parsed.path.lstrip("/"))
    base_path = f"{parsed.scheme}://{parsed.netloc}/{key_prefix}"
    filename = os.path.basename(parsed.path)
    store = from_url(url=base_path)
    file = open_reader(store=store, path=filename)
    return BackendArguments(
        filepath=filepath,
        file=file,
        object_reader=store,
    )

