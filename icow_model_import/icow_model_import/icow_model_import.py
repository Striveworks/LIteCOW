import logging

from argparse import ArgumentParser
from os import environ
from pathlib import Path
from typing import Dict, List, Optional, Union

import boto3
import torch

from mypy_boto3_s3 import S3Client


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
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
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
        net, out_file, dummy_forward_input, output_names, dynamic_axes,
    )


def onnx_file_to_s3(
    onnx_model: Union[str, Path],
    model_bucket: str,
    model_object_name: str,
    model_version: str,
) -> None:
    """Uploads a given onnx file to s3.

    Parameters
    ----------
    onnx_model : Union[str, Path]
        Path to the onnx model serialized to a file.
    model_bucket : str
        Bucket for the model to be uploaded to.
    model_object_name : str
        Name for the model to be used in s3.
    model_version : str
        Version tag for the uploaded object in s3.
    """

    # Validate the given model & model version in S3 and send to the model bucket
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
    )
    verify_model_version(model_bucket, model_object_name, model_version, s3_client)
    s3_client.upload_file(
        onnx_model, model_bucket, model_object_name,
    )
    s3_client.put_object_tagging(
        Bucket=model_bucket,
        Key=model_object_name,
        Tagging={"TagSet": [{"Key": "model-version", "Value": str(model_version)}]},
    )
    logging.info(
        "Model is available at s3://%s/%s with tag {'model-version': '%s'}",
        model_bucket,
        model_object_name,
        model_version,
    )


def cli() -> None:
    """Import a model from an existing onnx file into s3 for icow."""
    parser = ArgumentParser(
        description="Import a model from an existing onnx file into icow."
    )
    parser.add_argument(
        "onnx_model_path", help="Path to the serialized onnx model file."
    )
    parser.add_argument(
        "model_bucket", help="S3 bucket name for uploading the model to."
    )
    parser.add_argument(
        "model_object_name", help="Name for the uploaded s3 model object."
    )
    parser.add_argument(
        "model_version", help="Version tag added to the uploaded s3 model object."
    )
    args = parser.parse_args()
    onnx_file_to_s3(
        args.onnx_model_path,
        args.model_bucket,
        args.model_object_name,
        args.model_version,
    )
