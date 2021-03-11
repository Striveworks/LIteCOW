from typing import Optional, Union, List
from io import BytesIO
from json import loads

from PIL import Image
import numpy as np

from litecow_common.litecow_pb2 import ComputerVisionRequest
from litecow_common.litecow_pb2_grpc import ICOWStub

def decode_replacement_hook(obj):
    """Json object replacement hook that handles np.ndarrays.

    Json object replacement hook that handles np.ndarrays.
    Intended to be used with python's json.loads or json.load. Will be called as
    objects are loaded from the json, if the keys match np.ndarray then
    the respective type, np.ndarray will be loaded instead of
    the dict representation; otherwise it simply returns the passed dict.

    Parameters:
    -----------
    obj: dict
        A dict representing json loaded so far.

    Returns:
    --------
    Union[dict, np.ndarray]
    """
    array_key = "np.ndarray"

    if array_key in obj.keys():
        return np.array(obj[array_key])
    return obj

def to_image_bytes(image):
    with BytesIO() as image_file:
        image.save(image_file, "png")
        return image_file.getvalue()

class ICOWClient:
    def __init__(self, channel):
        self.stub = ICOWStub(channel)

    def get_cv_inference(self, model_key: str, images: Union[List[Image.Image], Image.Image], model_version: Optional[str] = None) -> np.ndarray:
        """ Get computer vision inference with images

        Parameters:
        -----------
        model_key: str
            The key to the model in s3.
        images: Union[List[PIL.Image.Image], PIL.Image.Image]
            The images to run inference on.
        model_version: Optional[str]
            The version of the model to use.

        Returns:
        --------
        numpy.ndarray
            Results of inference
        """
        # ensure images is a list of images
        images_list = images if isinstance(images, List) else [images]
        images_bytes = list(map(to_image_bytes, images_list))
        request_kwargs = locals()

        field_names = ComputerVisionRequest.DESCRIPTOR.fields_by_name.keys()
        filtered_kwargs = {key:value for key,value in request_kwargs.items() if key in field_names and value}
        return loads(self.stub.get_cv_inference(ComputerVisionRequest(**filtered_kwargs)).response, object_hook=decode_replacement_hook)
