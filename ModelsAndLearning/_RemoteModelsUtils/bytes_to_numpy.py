import numpy as np

def bytes_to_numpy(serialized_arr: str) -> np.array:
    sep = '|'.encode('utf-8')
    i_0 = serialized_arr.find(sep)
    i_1 = serialized_arr.find(sep, i_0 + 1)
    arr_dtype = serialized_arr[:i_0].decode('utf-8')
    arr_shape = tuple([int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
    arr_str = serialized_arr[i_1 + 1:]
    arr = np.frombuffer(arr_str, dtype = arr_dtype).reshape(arr_shape)
    return arr