from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from argparse import ArgumentParser
from os import environ
from urllib.parse import urlparse
from json import JSONEncoder, dumps
from io import BytesIO

import boto3
import grpc
from onnxruntime import InferenceSession
from PIL import Image
import numpy as np

from litecow_common.litecow_pb2_grpc import ICOWServicer, add_ICOWServicer_to_server
from litecow_common.litecow_pb2 import ResponseDict


class ResponseJSONEncoder(JSONEncoder):
    """Custom Json Encoder to np.ndarrays in json."""

    def default(self, o):
        """Returns a serializeable object for argument o or raises an exception if not serializable.

        Parameters:
        -----------
        o : obj
            Object to make serializable.

        Returns:
        --------
        dict
        """
        type_conv = {
            np.ndarray: lambda o: {"np.ndarray": o.tolist()},
        }
        return type_conv.get(type(o), super().default)(o)


def array_from_image(image_bytes):
    with BytesIO(image_bytes) as image_file:
        return Image.open(image_file).convert("RGB")


class ICOWServicer(ICOWServicer):
    def __init__(
        self,
        batch_hint: int,
        aws_access_key: str,
        aws_secret_key: str,
        endpoint_url: str,
    ):
        """Initializes ICOWService

        Parameters:
        -----------
        batch_hint: int
            Size to start batches at. Automatically reduced when overflow occurs.
        aws_access_key: string
            Access key for s3 storage
        aws_secret_key: string
            Secret key for s3 stroage
        endpoint_url: string
            Url for s3 storage
        """
        self.batch_hint = batch_hint
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            endpoint_url=endpoint_url,
        )

    def get_cv_inference(self, request, context):
        """Handles requests for service get_cv_inference calls.

        Parameters:
        -----------
        request: litecow_common.litecow_pb2.ComputerVisionRequest
            The ComputerVisionRequest to answer.
        context: grpc.ServicerContext
            The context for this request

        Returns:
        --------
        litecow_common.litecow_pb2.ResponseDict
            Inference is encoded as string json and returned as ResponseDict
        """
        # first retrieve model
        try:
            model = self._get_model_version_from_s3(
                request.model_key, request.model_version
            )
        except Exception as error:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Error occured while loading model from s3:\n{error}")
            raise error

        # run over the batch
        input_name = model.get_inputs()[0].name
        try:
            # TODO decide what to do on differently sized images
            # TODO fix batch size on memory issue
            image_batch = np.moveaxis(np.float32(np.stack(list(map(array_from_image, request.images_bytes)), 0) / 255.0), -1, 1)
            result = model.run(None, {input_name: image_batch})
        except Exception as error:
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details(f"Error occured while doing inference:\n{error}")
            raise error

        # convert result to json
        return ResponseDict(response=dumps(result, cls=ResponseJSONEncoder))

    def _get_model_version_from_s3(
        self, s3_path: str, model_version: Optional[str]
    ) -> InferenceSession:
        """Retrieves a model version from s3

        Parameters:
        -----------
        s3_path: str
            s3 object key for model to load should use s3 scheme like
            s3://models/facenet.onnx
        model_version: Optional[str]
            The optional version of the model to load
            empty is treated like None.

        Returns:
        --------
        onnxruntime.InferenceSession:
            The requested model loaded into an inference session
        """
        parsed_url = urlparse(s3_path)
        bucket_name, object_name = parsed_url.netloc, parsed_url.path

        if parsed_url.scheme != "s3":
            raise Exception("Model s3_path must follow s3 protocol")
        if not model_version:
            model_version = None

        def is_correct_version(obj_version) -> Optional[Tuple[str, dict]]:
            version_id = obj_version["VersionId"]
            # check if any of the object_tags contain {"model-version":model_version}
            object_tagging = self.s3_client.get_object_tagging(Bucket=bucket_name, Key=object_name, VersionId=version_id)["TagSet"]
            has_right_version = any(map(lambda tag: tag["Key"] == "model-version" and tag["Value"] == model_version, object_tagging))
            return (version_id, object_tagging) if has_right_version else None

        # resolve version of obj
        if model_version:
            try:
                correct_version, obj_meta = next(
                    filter(
                        None,
                        map(
                            is_correct_version,
                            self.s3_client.list_object_versions(
                                Bucket=bucket_name, Prefix=object_name
                            )["Versions"],
                        ),
                    )
                )
            except Exception as error:
                raise Exception(
                    f"Error occured while checking for object version:\n{error}"
                )

        # retrieve the object
        with NamedTemporaryFile() as temp_file:
            self.s3_client.download_fileobj(
                bucket_name,
                object_name,
                temp_file,
                ExtraArgs={"VersionId": correct_version} if model_version else None,
            )
            return InferenceSession(temp_file.name)


def serve(
    workers: int,
    port: int,
    batch_hint: int,
    aws_access_key: str,
    aws_secret_key: str,
    aws_endpoint_url: str,
) -> None:
    """Run the ICOWServicer with arguments passed.

    Parameter:
    ----------
    workers: int
        The number of max workers for the ThreadPoolExecutor running the server.
    port: int
        The port number to bind the service to.
    batch_hint: int
        The initial number of inputs to batch together in a single forward pass.
    aws_access_key: str
        The aws access key for the s3 storage to be used by the service.
    aws_secret_key: str
        The aws secret key for the s3 storage to be used by the service.
    aws_endpoint_url: str
        The url for the s3 storage used by the service includes scheme and port.
        Like http://my-s3-service.us:1337
    """
    server = grpc.server(ThreadPoolExecutor(max_workers=workers))
    add_ICOWServicer_to_server(
        ICOWServicer(batch_hint, aws_access_key, aws_secret_key, aws_endpoint_url),
        server,
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    server.wait_for_termination()


def main():
    """Run the icow_service with the arguments provided on the command line.

    Uses arguments provided on the command line and environment variables set on
    the system to run an icow_service. Must set environment variables: AWS_ACCESS_KEY
    AWS_SECRET_KEY, S3ENDPOINT_URL to values for the s3 storage so the icow_service
    can access s3.
    """
    parser = ArgumentParser(description="Run GRPC server for inference.")
    parser.add_argument(
        "num_worker", type=int, help="Number of thread workers for server."
    )
    parser.add_argument("port", type=int, help="Port number to serve on.")
    parser.add_argument(
        "batch_hint",
        type=int,
        nargs="?",
        default=16,
        help="Batch size inference should start with. Higher values may overflow the gpu or system memory and cause multiple retries to be necessary for an inference to complete, as the batch size is reduce by at each failure.",
    )
    args = parser.parse_args()
    serve(
        args.num_worker,
        args.port,
        args.batch_hint,
        environ["AWS_ACCESS_KEY"],
        environ["AWS_SECRET_KEY"],
        environ["S3ENDPOINT_URL"],
    )


if __name__ == "__main__":
    main()
