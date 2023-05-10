import io
import numpy as np
import torch as th

def tensor_to_byte(t):
    shape = t.shape
    if len(shape) == 3:
        b1 = int.to_bytes(shape[1], 2, byteorder='little')
        b2 = int.to_bytes(shape[2], 2, byteorder='little')
    else:
        b1 = int.to_bytes(shape[1], 2, byteorder='little')
        b2 = int.to_bytes(1, 2, byteorder='little')
    tb = t.detach().cpu().numpy().tobytes()
    b = b1 + b2 + tb
    return b

def get_next_tensor(b, dtype):
    b1 = int.from_bytes(b[:2], byteorder='little')
    b = b[2:]
    b2 = int.from_bytes(b[:2], byteorder='little')
    b = b[2:]
    num_bytes = b1 * b2 * (2 if dtype == np.float16 else 4)
    t = th.from_numpy(np.frombuffer(b[:num_bytes], dtype=dtype).copy())
    t = t.reshape(1, b1, b2)
    b = b[num_bytes:]
    return t, b

def get_bytestream(b):
    length = int.from_bytes(b[:4], byteorder='little')
    b = b[4:]
    return b[:length], b[length:]
