from typing import Dict, List

import numpy as np

from . import litecow_pb2


def prepare_array(array: np.ndarray) -> litecow_pb2.Array:
    array_type = str(array.dtype)
    array_cls = {
        "float64": ("double_array", litecow_pb2.DoubleArray),
        "float32": ("float_array", litecow_pb2.FloatArray),
        "int32": ("int32_array", litecow_pb2.Int32Array),
        "int64": ("int64_array", litecow_pb2.Int64Array),
    }
    f_name, cls = array_cls[array_type]
    kwargs = {"shape": array.shape, f_name: cls(values=array.flatten().tolist())}
    return litecow_pb2.Array(**kwargs)


def prepare_named_arrays(
    named_arrays: Dict[str, np.ndarray]
) -> litecow_pb2.NamedArrays:
    return litecow_pb2.NamedArrays(
        name_to_array={
            name: prepare_array(array) for name, array in named_arrays.items()
        }
    )


def prepare_array_list(arrays: List[np.ndarray]) -> litecow_pb2.ArrayList:
    return litecow_pb2.ArrayList(arrays=list(map(prepare_array, arrays)))

def unprepare_array(array:  litecow_pb2.Array) -> np.ndarray:
    if array.HasField('double_array'):
        np_array = np.float64(array.double_array.values)
    elif array.HasField('float_array'):
        np_array = np.float32(array.float_array.values)
    elif array.HasField('int32_array'):
        np_array = np.int32(array.int32_array.values)
    elif array.HasField('int64_array'):
        np_array = np.int64(array.int64_array.values)
    return np_array.reshape(array.shape)

def unprepare_named_arrays(named_arrays: litecow_pb2.NamedArrays) -> Dict[str, np.ndarray]:
    return {name:unprepare_array(array) for name, array in named_arrays.name_to_array.items()}

def unprepare_array_list(arrays: litecow_pb2.ArrayList) -> List[np.ndarray]:
    return list(map(unprepare_array, arrays.arrays))
