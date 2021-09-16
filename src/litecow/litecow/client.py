from typing import Optional, Union, List, Dict
from json import dumps

import grpc
import numpy as np

from litecow.common.litecow_pb2 import InferenceRequest
from litecow.common.litecow_pb2_grpc import ICOWStub
from litecow.common import common


class ICOWClient:
    @classmethod
    def create_with_channel_options(cls, endpoint, secure=False, credentials=None, options=None, compression=None):
        """Create grpc channel and injects in to the InferenceCowClient constructor

        Parameters:
        -----------
        endpoint: string
            Endpoint of ICOW service
        secure: bool
            Use a secure or insecure grpc channel
        credentials: grpc.ChannelCredentials:
            A ChannelCredentials instance.
        options:  List[tuple]
            An optional list of key-value pairs (channel_arguments in gRPC Core runtime) to configure the channel.
        compression: grpc.Compression
            An optional value indicating the compression method to be used over the lifetime of the grpc channel. This is an EXPERIMENTAL option.

        Returns:
        -----------
        InferenceCowClient
        """
        json_config = dumps(
            {
                "methodConfig": [
                    {
                        "name": [{"service": ".ICOW"}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "10s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    }
                ]
            }
        )
        default_options = [
            ("grpc.service_config", json_config),
            ("grpc.max_send_message_length", int(2e8)),
            ("grpc.max_receive_message_length", int(2e8)),
        ]

        # If duplicate keys are given GRPC will apply the first.
        # Add default options to back of options list  to allow users to override
        if options is None:
            options = default_options
        else:
            options.extend(default_options)

        if secure:
            if credentials is None:
                raise Exception("Must provide credentials when creating secure connection")
            channel = grpc.secure_channel(endpoint, credentials, options, compression)
        else:
            channel = grpc.insecure_channel(endpoint, options, compression)

        return cls(channel)

    def __init__(self, channel):
        """Initializes an ICOWClient

        Parameters
        ----------
        channel : grpc._channel.Channel
            The channel to connect with the server on

        Examples
        --------
        >>> ICOWClient(grpc.insecure_channel('localhost:8080'))
        """
        self.stub = ICOWStub(channel)

    def get_inference(
        self,
        model_key: str,
        inputs: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray],
        outputs: Optional[List[str]] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """ Get inference with input arrays

        Parameters
        ----------
        model_key : str
            The key to the model in s3.
        inputs : Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]
            The input(s) to run inference on may be a dict 
            mapping input names to arrays, a list of input arrays, or a single input array
        outputs : Optional[List[str]] = None
            The outputs of the inference to return, if
            None all will be returned.
        model_version : Optional[str]
            The version of the model to use.

        Returns
        -------
        Dict[str, np.ndarray]
            Results of inference

        Examples
        --------
        >>> client.get_inference("s3://models/my_model_key.onnx", np.float32(np.random.random((1, 3, 10, 10))), outputs=["loc", "landms"], model_version="v1")
        """
        # ensure inputs is formatted correctly
        if isinstance(inputs, Dict):
            named_inputs = common.prepare_named_arrays(inputs)
        elif isinstance(inputs, List):
            unnamed_inputs = common.prepare_array_list(inputs)
        else:
            unnamed_inputs = common.prepare_array_list([inputs])
        request_kwargs = locals()

        field_names = InferenceRequest.DESCRIPTOR.fields_by_name.keys()
        filtered_kwargs = {
            key: value
            for key, value in request_kwargs.items()
            if key in field_names and value
        }
        return common.unprepare_named_arrays(
            self.stub.get_inference(InferenceRequest(**filtered_kwargs))
        )
