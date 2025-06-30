"""
Icechunk repository management.

This module contains functions for creating and managing Icechunk repositories.
"""

import boto3
import icechunk
from config import bucket, directory, store_name


def open_or_create_repo():
    """
    Open or create an Icechunk repository.

    Returns:
        An Icechunk repository object
    """
    # Config for repo storage
    session = boto3.Session()

    # Get the credentials from the session
    credentials = session.get_credentials()

    # Extract the actual key, secret, and token
    creds = credentials.get_frozen_credentials()
    storage_config = icechunk.s3_storage(
        bucket=bucket,
        prefix=f"{directory}/{store_name}",
        region="us-west-2",
        access_key_id=creds.access_key,
        secret_access_key=creds.secret_key,
        session_token=creds.token,
    )

    # Config for repo
    repo_config = icechunk.RepositoryConfig.default()
    repo_config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(
            "s3", "s3://", icechunk.s3_store(region="us-west-2")
        )
    )

    # Config for repo virtual chunk credentials
    virtual_chunk_creds = icechunk.containers_credentials(
        s3=icechunk.s3_credentials(anonymous=False)
    )

    repo = icechunk.Repository.open_or_create(
        storage=storage_config,
        config=repo_config,
        virtual_chunk_credentials=virtual_chunk_creds,
    )
    return repo
