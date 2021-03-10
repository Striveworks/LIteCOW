from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator, Tuple, Union

import boto3
import onnxruntime
import pytest
import torch

from mypy_boto3_s3 import S3Client
from mypy_boto3_s3.service_resource import Bucket

from icow_model_import import (
    onnx_file_to_s3,
    pytorch_to_onnx_file,
    serialize_model_to_file,
    verify_model_version,
)


TEST_BUCKET_NAME = "models"


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
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
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
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
    )
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


# verify_model_version


def test_no_existing_objects_pos() -> None:
    """Test that verify_model_version won't raise any error if there are no objects by
    the name of the given model.
    """
    verify_model_version(TEST_BUCKET_NAME, "test_model", 0)


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
            tempfile,
            TEST_BUCKET_NAME,
            model_name,
            ExtraArgs={"Metadata": {"model-version": "1"}},
        )
    verify_model_version(TEST_BUCKET_NAME, model_name, 2)


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
            tempfile,
            TEST_BUCKET_NAME,
            model_name,
            ExtraArgs={"Metadata": {"model-version": "1"}},
        )
    with pytest.raises(ValueError) as err:
        verify_model_version(TEST_BUCKET_NAME, model_name, 1)
    assert "already exists" in str(err.value)


# onnx_file_to_s3


def test_single_model_upload_pos(boto_bucket: Tuple[Bucket, str]) -> None:
    """Test that onnx_file_to_s3 will correctly create a single model object with
    versioning metadata.

    Parameters
    ----------
    boto_bucket : Tuple[Bucket, str]
        boto3 S3 bucket and the bucket name.
    """
    # Upload the model
    boto_bucket, model_bucket = boto_bucket
    model_name = "simple_model"
    version = "my-version"
    onnx_file_to_s3(
        "tests/data/simple_onnx_model.onnx", model_bucket, model_name, version
    )
    models = list(boto_bucket.objects.all())
    assert len(models) == 1

    # Check version metadata added correctly
    versions = boto_bucket.object_versions.filter(Prefix=model_name)
    assert len(list(versions)) == 1
    obj = list(versions)[0].get()
    assert obj["Metadata"]["model-version"] == version


def test_multiple_model_versions_pos(boto_bucket: Tuple[Bucket, str]) -> None:
    """Test that onnx_file_to_s3 will correctly create multiple model objects of the
    same model with versioning metadata.

    Parameters
    ----------
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
        obj = version.get()
        assert obj["Metadata"]["model-version"] == str(version_int)
