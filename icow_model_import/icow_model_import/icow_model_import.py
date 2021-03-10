import logging

from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Union

import boto3
import torch


def serialize_model_to_file(
    out_path: Union[str, Path],
    net: torch.nn.Module,
    dummy_forward_input: torch.Tensor,
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
) -> None:
    """Serializes a given pytorch model to a file path.

    Parameters
    ----------
    out_path : Union[str, Path]
        Path where the serialized model should be written to.
    net : torch.nn.Module
        Pytorch model to serialize.
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
    model_bucket: str, model_name: str, model_version: int,
) -> None:
    """Verify that the given model and version do not conflict with pre-existing S3
    objects.

    Parameters
    ----------
    model_bucket : str
        Bucket that models are uploaded to.
    model_name : str
        Name of the model and bucket object.
    model_version : int
        New version for a model.

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
        bucket_object = version.get()
        if int(bucket_object["Metadata"]["model-version"]) == model_version:
            logging.error("Given already existing model version for serializing")
            raise ValueError(
                f"Given model version '{model_version}', already exists, please try "
                "again with a new version"
            )


def pytorch_to_bucket(
    net: torch.nn.Module,
    model_input_height: int,
    model_input_width: int,
    model_bucket: str,
    model_object_name: str,
    model_version: int,
    dynamic_shape: Optional[bool] = True,
    output_names: Optional[List[str]] = None,
    dummy_forward_input: Optional[torch.Tensor] = None,
) -> None:
    """Serialize a pytorch network with some given dummy input and dynamic batch sizes
    and export the model to an S3 bucket. The serialized model is done through tracing
    and/or scripting as shown in
    https://pytorch.org/docs/stable/onnx.html#tracing-vs-scripting. If the network's
    forward function uses control or loop structures, only some of these can be captured
    with tracing/scripting.

    Parameters
    ----------
    net : torch.nn.Module
        Network to serialize.
    model_input_height : int
        Model input dimension height.
    model_input_width : int
        Model input dimension width.
    model_bucket : str
        S3 bucket to use for storing the serialized models.
    model_object_name : str
        Name for the uploaded model object in s3.
    model_version : int
        Version for the uploaded model object.
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
    output_names = output_names or ["output"]
    net.eval()

    # Validate the given model and model version
    verify_model_version(model_bucket, model_object_name, model_version)

    # Create random tensor used for onnx export and run it through the model
    if dummy_forward_input is None:
        dummy_forward_input = torch.randn(
            1, 3, model_input_height, model_input_width
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    net(dummy_forward_input)

    # Export the model via onnx and send to the model bucket
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=environ["AWS_SECRET_KEY"],
        endpoint_url=environ["S3ENDPOINT_URL"],
    )
    if dynamic_shape:
        dynamic_axes = {"input": {0: "batch_size", 2: "height", 3: "width"}}
    else:
        dynamic_axes = {"input": {0: "batch_size"}}
    with NamedTemporaryFile() as tempfile:
        serialize_model_to_file(
            tempfile.name, net, dummy_forward_input, output_names, dynamic_axes,
        )
        s3_client.upload_fileobj(
            tempfile,
            model_bucket,
            f"{model_object_name}",
            ExtraArgs={"Metadata": {"model-version": str(model_version)}},
        )
    s3_path = f"s3://{model_bucket}/{model_object_name}"
    logging.info(f"Model is available at {s3_path}")
