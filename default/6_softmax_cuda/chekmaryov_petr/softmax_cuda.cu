#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

__global__ void SoftmaxKernel(const float* __restrict__ input, float* __restrict__ output, int row_size)
{
    extern __shared__ float scratch[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    const float* in_row = input + static_cast<long long>(row) * row_size;
    float* out_row = output + static_cast<long long>(row) * row_size;

    float local_max = -INFINITY;
    for (int j = tid; j < row_size; j += bdim)
        local_max = fmaxf(local_max, in_row[j]);

    scratch[tid] = local_max;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
        __syncthreads();
    }
    const float row_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < row_size; j += bdim)
    {
        const float e = __expf(in_row[j] - row_max);
        out_row[j] = e;
        local_sum += e;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float inv_sum = 1.0f / scratch[0];
    __syncthreads();


    for (int j = tid; j < row_size; j += bdim)
        out_row[j] *= inv_sum;
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count)
{
    if (input.empty() || row_count <= 0)
        return std::vector<float>();

    const int row_size = static_cast<int>(input.size() / row_count);
    const std::size_t elems = input.size();
    const std::size_t bytes = elems * sizeof(float);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);

    int block = 1;
    while (block < row_size && block < 1024)
        block <<= 1;
    const dim3 grid(static_cast<unsigned>(row_count));
    const dim3 threads(static_cast<unsigned>(block));
    const std::size_t smem = static_cast<std::size_t>(block) * sizeof(float);

    SoftmaxKernel<<<grid, threads, smem, stream>>>(d_in, d_out, row_size);

    std::vector<float> output(elems);
    cudaMemcpyAsync(output.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    return output;
}
