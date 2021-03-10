from os import environ

import boto3
import pytest

from mypy_boto3_s3 import S3Client

from icow_model_import.initialize_s3 import initialize_s3


# initialize_s3


@pytest.fixture(autouse=True)
def delete_all_buckets_but_models() -> None:
    """Deletes all buckets in a given S3 instance before and after a pytest, as long as
    there are no objects in them, except for a 'models' bucket.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
    )
    buckets = s3_client.list_buckets()["Buckets"]
    for bucket in buckets:
        if bucket["Name"] != "models":
            s3_client.delete_bucket(Bucket=bucket["Name"])

    yield

    buckets = s3_client.list_buckets()["Buckets"]
    for bucket in buckets:
        if bucket["Name"] != "models":
            s3_client.delete_bucket(Bucket=bucket["Name"])


def test_no_bucket_name_conflict_pos(boto_client: S3Client) -> None:
    """Test that initialize_s3 will create a bucket if the bucket does not exist in the
    s3 instance.

    Parameters
    ----------
    boto_client : S3Client
        Pytest fixture yielding a s3 client.
    """
    # Check bucket name not in list of buckets
    bucket_name = "non-conflict-name"
    buckets = boto_client.list_buckets()["Buckets"]
    for bucket in buckets:
        assert bucket["Name"] != bucket_name

    # Init bucket and check it's now in the bucket list with versioning enabled
    initialize_s3(bucket_name)
    buckets = boto_client.list_buckets()["Buckets"]
    in_buckets = False
    for bucket in buckets:
        if bucket["Name"] == bucket_name:
            in_buckets = True
            break
    assert in_buckets
    assert boto_client.get_bucket_versioning(Bucket=bucket_name)["Status"] == "Enabled"


def test_bucket_name_conflict_pos(boto_client: S3Client) -> None:
    """Test that initialize_s3 will create not create a bucket if the bucket does
    exist in the s3 instance.

    Parameters
    ----------
    boto_client : S3Client
        Pytest fixture yielding a s3 client.
    """
    # Create bucket
    bucket_name = "test-conflict"
    boto_client.create_bucket(Bucket=bucket_name)

    # Init bucket and check it's still in the bucket list
    initialize_s3(bucket_name)
    buckets = boto_client.list_buckets()["Buckets"]
    in_buckets = False
    for bucket in buckets:
        if bucket["Name"] == bucket_name:
            in_buckets = True
            break
    assert in_buckets


def test_bucket_name_conflict_enable_version_pos(boto_client: S3Client) -> None:
    """Test that initialize_s3 will create not create a bucket if the bucket does
    exist in the s3 instance and it will enable versioning.

    Parameters
    ----------
    boto_client : S3Client
        Pytest fixture yielding a s3 client.
    """
    # Create bucket and disable versioning on the bucket just made
    bucket_name = "test-conflict"
    boto_client.create_bucket(Bucket=bucket_name)
    boto_client.put_bucket_versioning(
        Bucket=bucket_name, VersioningConfiguration={"Status": "Suspended"},
    )

    # Init bucket and check it's in the bucket list and with versioning
    initialize_s3(bucket_name)
    buckets = boto_client.list_buckets()["Buckets"]
    in_buckets = False
    for bucket in buckets:
        if bucket["Name"] == bucket_name:
            in_buckets = True
            break
    assert in_buckets
    assert boto_client.get_bucket_versioning(Bucket=bucket_name)["Status"] == "Enabled"
