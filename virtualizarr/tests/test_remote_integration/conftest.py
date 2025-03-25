import time

import pytest


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


@pytest.fixture(scope="session")
def minio_bucket(container):
    # Setup with guidance from https://medium.com/@sant1/using-minio-with-docker-and-python-cbbad397cb5d
    from minio import Minio

    # Initialize MinIO client
    client = Minio(
        "localhost:9000",
        access_key=container["username"],
        secret_key=container["password"],
        secure=False,
    )
    client.make_bucket("mybucket")
    yield {
        "port": container["port"],
        "username": container["username"],
        "password": container["password"],
        "bucket": "mybucket",
    }
