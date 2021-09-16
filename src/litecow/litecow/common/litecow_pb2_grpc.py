# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from litecow.common import litecow_pb2 as litecow_dot_common_dot_litecow__pb2


class ICOWStub(object):
    """Inference with Collected Onnx Weights service definition 
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.get_inference = channel.unary_unary(
                '/ICOW/get_inference',
                request_serializer=litecow_dot_common_dot_litecow__pb2.InferenceRequest.SerializeToString,
                response_deserializer=litecow_dot_common_dot_litecow__pb2.NamedArrays.FromString,
                )


class ICOWServicer(object):
    """Inference with Collected Onnx Weights service definition 
    """

    def get_inference(self, request, context):
        """get inference
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ICOWServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'get_inference': grpc.unary_unary_rpc_method_handler(
                    servicer.get_inference,
                    request_deserializer=litecow_dot_common_dot_litecow__pb2.InferenceRequest.FromString,
                    response_serializer=litecow_dot_common_dot_litecow__pb2.NamedArrays.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ICOW', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ICOW(object):
    """Inference with Collected Onnx Weights service definition 
    """

    @staticmethod
    def get_inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ICOW/get_inference',
            litecow_dot_common_dot_litecow__pb2.InferenceRequest.SerializeToString,
            litecow_dot_common_dot_litecow__pb2.NamedArrays.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)