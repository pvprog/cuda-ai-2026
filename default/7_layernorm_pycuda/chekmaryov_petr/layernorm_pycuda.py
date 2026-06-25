import math
import numpy as np
from numba import cuda, float32

_MAX_BLOCK = 1024

@cuda.jit(cache=True)
def _layernorm_kernel(x, gamma, beta, y, row_size, eps):
    sdata = cuda.shared.array(_MAX_BLOCK, dtype=float32)

    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    base = row * row_size

    s = float32(0.0)
    j = tid
    while j < row_size:
        s += x[base + j]
        j += bdim
    sdata[tid] = s
    cuda.syncthreads()
    k = bdim // 2
    while k > 0:
        if tid < k:
            sdata[tid] += sdata[tid + k]
        cuda.syncthreads()
        k //= 2
    mean = sdata[0] / float32(row_size)
    cuda.syncthreads()

    v = float32(0.0)
    j = tid
    while j < row_size:
        d = x[base + j] - mean
        v += d * d
        j += bdim
    sdata[tid] = v
    cuda.syncthreads()
    k = bdim // 2
    while k > 0:
        if tid < k:
            sdata[tid] += sdata[tid + k]
        cuda.syncthreads()
        k //= 2
    var = sdata[0] / float32(row_size)
    inv_std = float32(1.0) / math.sqrt(var + eps)
    cuda.syncthreads()

    j = tid
    while j < row_size:
        y[base + j] = (x[base + j] - mean) * inv_std * gamma[j] + beta[j]
        j += bdim

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    row_size = int(row_size)

    x_in = np.asarray(input, dtype=np.float32)
    out_shape = x_in.shape
    x = np.ascontiguousarray(x_in.ravel(), dtype=np.float32)
    g = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32).ravel())
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).ravel())

    if row_size <= 0 or x.size == 0:
        return x.reshape(out_shape)

    rows = x.size // row_size

    block = 1
    while block < row_size and block < _MAX_BLOCK:
        block <<= 1

    d_x = cuda.to_device(x)
    d_g = cuda.to_device(g)
    d_b = cuda.to_device(b)
    d_y = cuda.device_array_like(x)

    _layernorm_kernel[rows, block](d_x, d_g, d_b, d_y, np.int32(row_size), np.float32(eps))

    y = d_y.copy_to_host()
    return y.reshape(out_shape)