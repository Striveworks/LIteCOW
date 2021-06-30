import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Tuple, Union
from urllib.parse import urlparse

import boto3
import pytest
import torch
from botocore.stub import Stubber

import onnxruntime
from litecow_models.model import (ModelLoader, convert_github_raw,
                                  create_s3_client, download_file,
                                  initialize_s3, pytorch_to_onnx_file,
                                  serialize_model_to_file,
                                  verify_model_version)
from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.service_resource import Bucket

TEST_BUCKET_NAME = "models"


def test_convert_github_raw():
    source = "https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx"
    scheme, netloc, path, _, _, _ = urlparse(source)
    assert (
        convert_github_raw(scheme, netloc, path)
        == "https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx"
    )


def test_create_s3_client():
    client = create_s3_client()
    stubber = Stubber(client)
    list_buckets_response = {
        "Owner": {"DisplayName": "name", "ID": "EXAMPLE123"},
        "Buckets": [{"CreationDate": "2016-05-25T16:55:48.000Z", "Name": "foo"}],
    }

    expected_params = {}
    stubber.add_response("list_buckets", list_buckets_response, expected_params)

    with stubber:
        response = client.list_buckets()

    assert response == list_buckets_response


def test_download_file():
    filename = download_file("https://github.com/onnx/models/blob/master/README.md")
    assert os.path.exists(filename)
    os.remove(filename)



@pytest.fixture(autouse=True)
def delete_all_buckets_but_models() -> None:
    """Deletes all buckets in a given S3 instance before and after a pytest, as long as
    there are no objects in them, except for a 'models' bucket.
    """
    s3_client = create_s3_client()
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


    class SimpleMLP(torch.nn.Module):
        """Simple neural network for testing purposes.

        Parameters
        ----------
        layer_sizes : list
            A list of the layer sizes inclusive of the input
            and output layers
        classification : bool
            If True, softmax is applied to the output. Otherwise,
            L2 normalization is performed.
        """

        def __init__(
            self, layer_sizes=[256, 128, 8], classification=True,
        ):
            super().__init__()

            self.classification = classification
            modules = []
            for i, s in enumerate(layer_sizes[1:]):
                modules += [torch.nn.Linear(layer_sizes[i], s)]
                if i + 1 != len(layer_sizes) - 1:
                    modules += [torch.nn.ReLU()]
            self.net = torch.nn.Sequential(*modules)

        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            """Function for einputecuting the forward pass of a torch nn model.

            Parameters
            ----------
            input_tensor : torch.Tensor
                Input to the model.

            Returns
            -------
            torch.tensor
                Result of the forward pass of the network.
            """
            input_tensor = input_tensor.reshape(
                input_tensor.shape[0], input_tensor.shape[-1]
            )
            input_tensor = self.net(input_tensor)
            if not self.classification:
                input_tensor = torch.nn.functional.normalize(input_tensor)
            return input_tensor


    # serialize_model_to_file


    def test_serialize_to_temp_file_pos() -> None:
        """Test that serialize_model_to_file will correctly serialize and deserialize from
        temporary file.
        """
        # Serialize a simple model to a tempfile
        model = SimpleMLP()
        dummy_forward_input = torch.randn(1, 1, 256).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with NamedTemporaryFile() as tempfile:
            serialize_model_to_file(
                model,
                tempfile.name,
                dummy_forward_input,
                ["output"],
                {"input": {0: "batch_size"}},
            )
            tempfile.seek(0)

            # Deserialize the model from the tempfile
            onnx_options = onnxruntime.SessionOptions()
            onnxruntime.InferenceSession(
                str(tempfile.name), sess_options=onnx_options,
            )


    @pytest.fixture
    def boto_bucket() -> Iterator[Tuple[Bucket, str]]:
        """Pytest fixture that yields an S3 bucket for testing and the name of that bucket.

        Yields
        -------
        Iterator[Tuple[Bucket, str]]
            The s3 bucket and the name of the bucket.
        """
        s3 = boto3.resource(
            "s3",
            endpoint_url=os.getenv("S3_URL", "http://localhost:9000"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
            config=botocore.config.Config(signature_version="s3v4"),
            region_name=os.getenv("S3_REGION", "us-east-1"),
            verify=False,
        )
        yield s3.Bucket(TEST_BUCKET_NAME), TEST_BUCKET_NAME


    # pytorch_to_onnx_file


    def delete_all_object_versions(bucket: Bucket, s3_client: S3Client) -> None:
        """Deletes every object and every object's versions inside of an s3 bucket.

        Parameters
        ----------
        bucket : Bucket
            S3 bucket to delete objects from.
        s3_client : S3Client
            S3 client to delete objects with.
        """
        model_objects = bucket.objects.all()
        for model_object in model_objects:
            versions = bucket.object_versions.filter(Prefix=model_object.key)
            for version in versions:
                s3_client.delete_object(
                    Bucket=TEST_BUCKET_NAME, Key=model_object.key, VersionId=version.id,
                )


    @pytest.fixture(autouse=True)
    def clean_models() -> None:
        """Pytest fixture for clearing the test models after running a test function."""
        s3_client = create_s3_client()
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=environ["AWS_SECRET_KEY"],
            endpoint_url=environ["S3ENDPOINT_URL"],
        )
        bucket = s3.Bucket(TEST_BUCKET_NAME)
        delete_all_object_versions(bucket, s3_client)
        yield
        delete_all_object_versions(bucket, s3_client)


    def deserialize_model_from_file(file_path: Union[str, Path]) -> None:
        """Deserializes an onnx model from a file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the onnx model to deserialize.
        """
        onnx_options = onnxruntime.SessionOptions()
        onnxruntime.InferenceSession(
            file_path, sess_options=onnx_options,
        )


    def test_to_file_pos() -> None:
        """Test that pytorch_to_bucket will correctly serialize a model to file."""
        # Serialize and check the model can be deserialized after
        model = SimpleMLP()
        dummy_forward_input = torch.randn(1, 1, 256).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        with NamedTemporaryFile() as tempfile:
            pytorch_to_onnx_file(
                model,
                tempfile.name,
                1,
                256,
                dynamic_shape=False,
                dummy_forward_input=dummy_forward_input,
            )
            deserialize_model_from_file(tempfile.name)



    def test_no_existing_objects_pos(boto_client: S3Client) -> None:
        """Test that verify_model_version won't raise any error if there are no objects by
        the name of the given model.

        Parameters
        ----------
        boto_client : S3Client
            Client to use for uploading new objects.
        """
        verify_model_version(TEST_BUCKET_NAME, "test_model", 0, boto_client)


    def test_no_conflict_pos(boto_client: S3Client) -> None:
        """Test that verify_model_version will not raise an error if a model already exists
        in S3, but does not have a conflicting version.

        Parameters
        ----------
        boto_client : S3Client
            Client to use for uploading new objects.
        """
        model_name = "test_model"
        with NamedTemporaryFile() as tempfile:
            tempfile.write(b"test data")
            boto_client.upload_fileobj(
                tempfile, TEST_BUCKET_NAME, model_name,
            )
            version = 1
            boto_client.put_object_tagging(
                Bucket=TEST_BUCKET_NAME,
                Key=model_name,
                Tagging={"TagSet": [{"Key": "model-version", "Value": str(version)}]},
            )
        verify_model_version(TEST_BUCKET_NAME, model_name, 2, boto_client)


    def test_conflicting_versions_neg(boto_client: S3Client) -> None:
        """Test that verify_model_version will raise an error if a model and given version
        already exists in S3.

        Parameters
        ----------
        boto_client : S3Client
            Client to use for uploading new objects.
        """
        model_name = "test_model"
        with NamedTemporaryFile() as tempfile:
            tempfile.write(b"test data")
            boto_client.upload_fileobj(
                tempfile, TEST_BUCKET_NAME, model_name,
            )
            version = 2
            boto_client.put_object_tagging(
                Bucket=TEST_BUCKET_NAME,
                Key=model_name,
                Tagging={"TagSet": [{"Key": "model-version", "Value": str(version)}]},
            )
        with pytest.raises(ValueError) as err:
            verify_model_version(TEST_BUCKET_NAME, model_name, version, boto_client)
        assert "already exists" in str(err.value)



    def test_multiple_model_versions_pos(
        boto_client: S3Client, boto_bucket: Tuple[Bucket, str]
    ) -> None:
        """Test that onnx_file_to_s3 will correctly create multiple model objects of the
        same model with versioning metadata.

        Parameters
        ----------
        boto_client : S3Client
            Client to use for uploading new objects.
        boto_bucket : Tuple[Bucket, str]
            boto3 S3 bucket and the bucket name.
        """
        # Upload the same model twice
        boto_bucket, model_bucket = boto_bucket
        model_name = "simple_model"
        for version in range(2):
            onnx_file_to_s3(
                "tests/data/simple_onnx_model.onnx", model_bucket, model_name, version
            )
        models = list(boto_bucket.objects.all())
        assert len(models) == 1

        # Check version metadata added correctly
        versions = boto_bucket.object_versions.filter(Prefix=model_name)
        assert len(list(versions)) == 2
        for version, version_int in zip(versions, [1, 0]):
            tags = boto_client.get_object_tagging(
                Bucket=model_bucket, Key=model_name, VersionId=version.id
            )["TagSet"]
            for tag in tags:
                if tag["Key"] == "model-version":
                    assert tag["Value"] == str(version_int)
