import time

import numpy as np
import pytest
from xarray import Dataset
from xarray.core.variable import Variable

from conftest import ARRAYBYTES_CODEC, ZLIB_CODEC
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.tests import requires_minio


@pytest.fixture
def vds_with_manifest_arrays(array_v3_metadata) -> Dataset:
    arr = ManifestArray(
        chunkmanifest=ChunkManifest(
            entries={"0.0": dict(path="/test.nc", offset=6144, length=48)}
        ),
        metadata=array_v3_metadata(
            shape=(2, 3),
            data_type=np.dtype("<i8"),
            chunks=(2, 3),
            codecs=[ARRAYBYTES_CODEC, ZLIB_CODEC],
            fill_value=0,
        ),
    )
    var = Variable(dims=["x", "y"], data=arr, attrs={"units": "km"})
    return Dataset({"a": var}, attrs={"something": 0})


@requires_minio
@pytest.fixture(scope="session")
def container():
    import docker

    client = docker.from_env()
    port = 9000
    minio_container = client.containers.run(
        "quay.io/minio/minio",
        "server /data",
        detach=True,
        ports={f"{port}/tcp": port},
        environment={
            "MINIO_ACCESS_KEY": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
        },
    )
    time.sleep(3)  # give it time to boot
    # enter
    yield {
        "port": port,
        "endpoint": f"http://localhost:{port}",
        "username": "minioadmin",
        "password": "minioadmin",
    }
    # exit
    minio_container.stop()
    minio_container.remove()


@requires_minio
@pytest.fixture(scope="session")
def minio_bucket(container):
    # Setup with guidance from https://medium.com/@sant1/using-minio-with-docker-and-python-cbbad397cb5d
    from minio import Minio

    bucket = "mybucket"
    filename = "test.nc"
    # Initialize MinIO client
    client = Minio(
        "localhost:9000",
        access_key=container["username"],
        secret_key=container["password"],
        secure=False,
    )
    client.make_bucket(bucket)
    yield {
        "port": container["port"],
        "endpoint": container["endpoint"],
        "username": container["username"],
        "password": container["password"],
        "bucket": bucket,
        "file": filename,
        "client": client,
    }
