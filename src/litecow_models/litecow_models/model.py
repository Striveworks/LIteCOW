import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import boto3
import botocore
import requests
import torch
from tqdm import tqdm

from mypy_boto3_s3 import S3Client


class ModelLoader:
    @staticmethod
    def import_model(source: str, model_bucket: str, model_name: str, model_version: str) -> None:
        """Import model into model registry

        Parameters
        ----------
        source: str
            Source URL of model
        model_bucket: str
            Model registry bucket name
        model_name: str
            Name of model
        model_version : str
            Version tag for the uploaded object in s3.
        """
        scheme, netloc, path, _, _, _ = urlparse(source)
        if netloc == "github.com":
            source = convert_github_raw(scheme, netloc, path)
        if scheme in ("http", "https"):
            filename = download_file(source)
        else:
            filename = source
        client = create_s3_client()
        verify_model_version(model_bucket, model_name, model_version, client)
        client.upload_file(filename, model_bucket, model_name)
        client.put_object_tagging(
            Bucket=model_bucket,
            Key=model_name,
            Tagging={"TagSet": [{"Key": "model-version", "Value": str(model_version)}]},
        )
        logging.info(
            "Model is available at s3://%s/%s with tag {'model-version': '%s'}",
            model_bucket,
            model_name,
            model_version,
        )

    @staticmethod
    def export_model(model_bucket: str, model_name: str) -> None:
        """Export model from model registry

        Parameters
        ----------
        model_bucket : str
            Model registry bucket name
        model_name : str
            Name of model
        """
        client = create_s3_client()
        client.download_file(model_bucket, model_name, model_name)



def serialize_model_to_file(
    net: torch.nn.Module,
    out_path: Union[str, Path],
    dummy_forward_input: torch.Tensor,
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
) -> None:
    """Serializes a given pytorch model to a file path. Main use for this is verifying
    tempfile IOs work with serialization and deserialization of onnx models.

    Parameters
    ----------
    net : torch.nn.Module
        Pytorch model to serialize.
    out_path : Union[str, Path]
        Path where the serialized model should be written to.
    dummy_forward_input : torch.Tensor
        Forward input for the model to be used for tracing the model and creating the
        onnx execution graph for serialization.
    output_names : List[str]
        Name of the outputs for the model.
    dynamic_axes : Dict[str, Dict[int, str]]
        Dynamic axes configuration dictionary for onnx export.
    """
    torch.onnx.export(
        net.module if isinstance(net, torch.nn.DataParallel) else net,
        dummy_forward_input,
        out_path,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=True,
    )


def verify_model_version(
    model_bucket: str, model_name: str, model_version: str, s3_client: S3Client
) -> None:
    """Verify that the given model and version do not conflict with pre-existing S3
    objects.

    Parameters
    ----------
    model_bucket : str
        Bucket that models are uploaded to.
    model_name : str
        Name of the model and bucket object.
    model_version : str
        New version for a model.
    s3_client : S3Client
        S3 client for connecting to s3.

    Raises
    ------
    ValueError
        If the given model and version already exist in S3.
    """
    bucket = boto3.resource(
        "s3",
        endpoint_url=os.getenv("S3_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=botocore.config.Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        verify=False,
    ).Bucket(model_bucket)
    for version in bucket.object_versions.filter(Prefix=model_name):
        tags = s3_client.get_object_tagging(
            Bucket=model_bucket, Key=model_name, VersionId=version.id
        )["TagSet"]
        for tag in tags:
            if tag["Key"] == "model-version" and tag["Value"] == str(model_version):
                logging.error("Given already existing model version for serializing")
                raise ValueError(
                    f"Given model version '{model_version}', already exists, please "
                    "try again with a new version"
                )


def pytorch_to_onnx_file(
    net: torch.nn.Module,
    out_file: Union[str, Path],
    model_input_height: int,
    model_input_width: int,
    dynamic_shape: Optional[bool] = True,
    output_names: Optional[List[str]] = None,
    dummy_forward_input: Optional[torch.Tensor] = None,
) -> None:
    """Serialize a pytorch network with some given dummy input and dynamic batch sizes.
    The serialized model is done through tracing and/or scripting as shown in
    https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting. If the network's
    forward function uses control or loop structures, only some of these can be captured
    with tracing/scripting.

    Parameters
    ----------
    net : torch.nn.Module
        Network to serialize.
    out_file : Union[str, Path]
        File path to export the onnx file to.
    model_input_height : int
        Model input dimension height.
    model_input_width : int
        Model input dimension width.
    dynamic_shape : Optional[bool]
        Whether or not the spatial dimensions of the input are dynamic. Defaults to
        False, meaning the shape is static.
    output_names : Optional[List[str]]
        List of output headnames for mutli-head classification networks. Defaults to
        None which will be assigned to ["output"].
    dummy_forward_input : Optional[torch.Tensor]
        Dummy forward pass input that will be run through the model for tracing export
        purposes. Defaults to None meaning a dummy input will be generated.
    """
    # Set the network to evaluation mode
    output_names = output_names or ["output"]
    net.eval()

    # Create random tensor used for onnx export and run it through the model
    if dummy_forward_input is None:
        dummy_forward_input = torch.randn(
            1, 3, model_input_height, model_input_width
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    net(dummy_forward_input)

    # Serialize model
    if dynamic_shape:
        dynamic_axes = {"input": {0: "batch_size", 2: "height", 3: "width"}}
    else:
        dynamic_axes = {"input": {0: "batch_size"}}
    serialize_model_to_file(
        net,
        out_file,
        dummy_forward_input,
        output_names,
        dynamic_axes,
    )



def convert_github_raw(scheme: str, netloc: str, path: str) -> str:
    """Convert github blob URL to raw content URL

    Parameters
    ----------
    scheme: str
        URL scheme
    netloc: str
        Network host address
    path: str
        Path to file
    """
    return f"{scheme}://{netloc}{path.replace('blob', 'raw')}"


def create_s3_client() -> botocore.client:
    """Create boto3 S3 client"""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=botocore.config.Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        verify=False,
    )


def download_file(url: str) -> str:
    """Convert github blob URL to raw content URL

    Parameters
    ----------
    url: str
        Source URL of download

    Returns
    -------
    local_filename: str
        Local filename of downloaded file
    """
    local_filename = f"/tmp/{url.split('/')[-1]}"
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(local_filename, "wb") as file, tqdm(
        desc=local_filename,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return local_filename


def initialize_s3(bucket_name: str) -> None:
    """Connects to S3, insures that a bucket exists and it has versioning
    enabled.

    Parameters
    ----------
    bucket_name : str
        Name of the s3 bucket to init.
    """
    # Create a client for interaction with the S3 server
    s3_client = create_s3_client()

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
