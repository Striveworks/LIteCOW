from os import environ
from typing import Iterator

import boto3
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
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
    )
