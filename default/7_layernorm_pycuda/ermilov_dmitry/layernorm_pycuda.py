import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

kernel = SourceModule(r"""
#include <math.h>

#define BLOCK_SIZE 256

__global__ void LayerNormKernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int row_size,
    float eps)
{
    __shared__ float sdata[BLOCK_SIZE];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * row_size;

    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        local_sum += input[base + i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / row_size;
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float d = input[base + i] - mean;
        local_var += d * d;
    }

    sdata[tid] = local_var;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float inv_std = rsqrtf((sdata[0] / row_size) + eps);

    for (int i = tid; i < row_size; i += BLOCK_SIZE) {
        float x = input[base + i];
        output[base + i] = (x - mean) * inv_std * gamma[i] + beta[i];
    }
}
""")

_layernorm_kernel = kernel.get_function("LayerNormKernel")


def layernorm_pycuda(input_array, gamma, beta, row_size, eps=1e-5):
    input_array = np.asarray(input_array, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    row_count = input_array.size // row_size
    output = np.empty_like(input_array)

    d_input = cuda.mem_alloc(input_array.nbytes)
    d_gamma = cuda.mem_alloc(gamma.nbytes)
    d_beta = cuda.mem_alloc(beta.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_array, stream)
    cuda.memcpy_htod_async(d_gamma, gamma, stream)
    cuda.memcpy_htod_async(d_beta, beta, stream)

    _layernorm_kernel(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(int(row_count), 1, 1),
    )

    cuda.memcpy_dtoh_async(output, d_output, stream)

    stream.synchronize()

    d_input.free()
    d_gamma.free()
    d_beta.free()
    d_output.free()

    return output
