import logging
import os

from minio import Minio
from minio.commonconfig import ENABLED
from minio.versioningconfig import VersioningConfig


def main():
    """Connects to minio, insures that the model bucket exists and it has versioning
    enabled
    """
    # Create a client with the MinIO server
    client = Minio(
        os.environ["S3ENDPOINT_URL"],
        access_key=os.environ["AWS_ACCESS_KEY"],
        secret_key=os.environ["AWS_SECRET_KEY"],
        secure=False,
    )

    # Make 'model' bucket if it doesn't exist
    model_bucket = "models"
    if not client.bucket_exists(model_bucket):
        client.make_bucket(model_bucket)
        logging.info("Created bucket '%s'", model_bucket)
    else:
        logging.info("Bucket '%s' already exists", model_bucket)

    # Enable object versioning on the bucket
    client.set_bucket_versioning(model_bucket, VersioningConfig(ENABLED))
    logging.info("'%s' bucket versioning enabled", model_bucket)


if __name__ == "__main__":
    main()
