import logging
import os

from argparse import ArgumentParser

import boto3


def initialize_s3(bucket_name: str) -> None:
    """Connects to S3, insures that a bucket exists and it has versioning
    enabled.

    Parameters
    ----------
    bucket_name : str
        Name of the s3 bucket to init.
    """
    # Create a client for interaction with the S3 server
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        endpoint_url=os.environ["S3ENDPOINT_URL"],
    )

    # Make the bucket if it doesn't exist
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        logging.info("Created bucket '%s'", bucket_name)
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        logging.info("Bucket '%s' already exists", bucket_name)

    # Enable object versioning on the bucket
    s3_client.put_bucket_versioning(
        Bucket=bucket_name, VersioningConfiguration={"Status": "Enabled"},
    )
    logging.info("'%s' bucket versioning enabled", bucket_name)


def cli() -> None:
    """CLI command for parsing arguments to feed the initialize_s3 function."""
    parser = ArgumentParser(
        description="Initialize an s3 bucket with versioning enabled."
    )
    parser.add_argument("bucket_name", help="Name of s3 bucket to init.")
    args = parser.parse_args()
    initialize_s3(args.bucket_name)
