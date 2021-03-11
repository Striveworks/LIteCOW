import os
from typing import Iterator

import boto3
import botocore

import pytest

from mypy_boto3_s3 import S3Client


@pytest.fixture
def boto_client() -> Iterator[S3Client]:
    """Pytest fixture that yields an S3 client for testing.

    Yields
    -------
    Iterator[S3Client]
        The s3 client.
    """
    yield boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=botocore.config.Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        verify=False,
    )
