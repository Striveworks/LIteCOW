from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
from os import environ

import grpc

from litecow_common.litecow_pb2_grpc import ICOWServicer, add_ICOWServicer_to_server

class ICOWServicer(ICOWServicer):
    def __init__(self, batch_hint, aws_access_key, aws_secret_key, endpoint_url):
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
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.endpoint_url = endpoint_url

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
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def serve(workers: int, port: int, batch_hint: int, aws_access_key: str, aws_secret_key: str, aws_endpoint_url: str) -> None:
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
        ICOWServicer(
            batch_hint, aws_access_key, aws_secret_key, aws_endpoint_url
        ),
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
