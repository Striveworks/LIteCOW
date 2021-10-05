from typing import Any, Dict, List

import numpy as np

from . import litecow_pb2


def prepare_array(array: np.ndarray) -> litecow_pb2.Array:
    """Convert a numpy array to a litecow array

    Parameters
    ----------
    array: np.ndarray
        Array to convert

    Returns
    -------
    litecow.common.litecow_pb2.Array
    """
    try:
        array_type = str(array.dtype)
        array_cls = {
            "float64": ("double_array", litecow_pb2.DoubleArray),
            "float32": ("float_array", litecow_pb2.FloatArray),
            "int32": ("int32_array", litecow_pb2.Int32Array),
            "int64": ("int64_array", litecow_pb2.Int64Array),
            "object": ("dict_array", litecow_pb2.DictionaryArray),
        }
        f_name, cls = array_cls[array_type]
        if f_name == "dict_array":
            dict_array = []
            for dict in array:
                dict_proto = litecow_pb2.Dictionary(
                    pairs=[
                        litecow_pb2.Pair(
                            key=litecow_pb2.Key(**{key.__class__.__name__: key}),
                            value=litecow_pb2.Value(
                                **{value.__class__.__name__: value}
                            ),
                        )
                        for key, value in dict.items()
                    ]
                )
                dict_array.append(dict_proto)
            return litecow_pb2.Array(
                **{
                    "shape": array.shape,
                    f_name: litecow_pb2.DictionaryArray(values=dict_array),
                }
            )

        else:
            kwargs = {
                "shape": array.shape,
                f_name: cls(values=array.flatten().tolist()),
            }
            return litecow_pb2.Array(**kwargs)
    except Exception as e:
        raise Exception(
            "Unable to prepare array, input or output type is likely not supported by LItecow protobufs."
        ) from e


def prepare_named_objects(named_arrays: Dict[str, Any]) -> litecow_pb2.NamedArrays:
    """Convert a dictionary of arrays to a litecow NamedArrays

    Parameters
    ----------
    named_arrays: Dict[str, Any]
        Dictionary mapping string names to numpy arrays

    Returns
    -------
    litecow.common.litecow_pb2.NamedArrays
    """
    ret = litecow_pb2.NamedArrays(
        name_to_array={
            name: (
                prepare_array(obj)
                if isinstance(obj, np.ndarray)
                else prepare_array(np.array(obj))
            )
            for name, obj in named_arrays.items()
        }
    )

    return ret


def prepare_array_list(arrays: List[np.ndarray]) -> litecow_pb2.ArrayList:
    """Convert a list of numpy arrays to a litecow ArrayList

    Parameters
    ----------
    arrays: List[np.ndarray]
        Arrays to convert to litecow ArrayList

    Returns
    -------
    litecow.common.litecow_pb2.ArrayList
    """
    return litecow_pb2.ArrayList(arrays=list(map(prepare_array, arrays)))


def unprepare_array(array: litecow_pb2.Array) -> np.ndarray:
    """Convert a litecow Array to a numpy array

    Parameters
    ----------
    array: litecow.common.litecow_pb2.Array
        Array to convert back to numpy array

    Returns
    -------
    np.ndarray
    """
    if array.HasField("double_array"):
        np_array = np.float64(array.double_array.values)
    elif array.HasField("float_array"):
        np_array = np.float32(array.float_array.values)
    elif array.HasField("int32_array"):
        np_array = np.int32(array.int32_array.values)
    elif array.HasField("int64_array"):
        np_array = np.int64(array.int64_array.values)
    elif array.HasField("dict_array"):
        np_array = np.object_(
            [
                {
                    getattr(pair.key, pair.key.WhichOneof("key")): getattr(
                        pair.value, pair.value.WhichOneof("val")
                    )
                    for pair in d.pairs
                }
                for d in array.dict_array.values
            ]
        )

    return np_array.reshape(array.shape)


def unprepare_named_arrays(
    named_arrays: litecow_pb2.NamedArrays,
) -> Dict[str, np.ndarray]:
    """Convert litecow NamedArrays back to a dictionary of numpy arrays

    Parameters
    ----------
    named_arrays: litecow.common.litecow_pb2.NamedArrays
        NamedArrays to convert

    Returns
    -------
    Dict[str, np.ndarray]
    """
    return {
        name: unprepare_array(array)
        for name, array in named_arrays.name_to_array.items()
    }


def unprepare_array_list(arrays: litecow_pb2.ArrayList) -> List[np.ndarray]:
    """Convert litecow ArrayList back to a list of numpy arrays

    Parameters
    ----------
    arrays: litecow.common.litecow_pb2.ArrayList
        ArrayList to convert.

    Returns
    -------
    List[np.ndarray]
    """
    return list(map(unprepare_array, arrays.arrays))
