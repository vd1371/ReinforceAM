import numpy as np

def numpy_to_bytes(arr: np.array) -> str:
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return
