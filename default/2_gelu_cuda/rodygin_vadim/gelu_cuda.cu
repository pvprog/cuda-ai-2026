#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include "gelu_cuda.h"
#include <cmath>

__global__ void geluKernel(const float* __restrict__ d_in,
                           float*       __restrict__ d_out,
                           size_t n)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float x = d_in[idx];

    const float c0 = sqrt(2.0f / M_PI);
    const float c1 = 0.044715f;
    const float x3 = x * x * x;
    const float inner = c0 * (x + c1 * x3);
    const float tanh_inner = tanhf(inner);

    d_out[idx] = 0.5f * x * (1.0f + tanh_inner);
}

std::vector<float> GeluCUDA(const std::vector<float>& input)
{
    if (input.empty())
        return {};

    const size_t N = input.size();
    const size_t bytes = N * sizeof(float);
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    float* d_in  = nullptr;
    float* d_out = nullptr;
    
    err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to allocate device memory for input");
    }
    
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to allocate device memory for output");
    }

    err = cudaMemcpyAsync(d_in, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy data to device");
    }

    float* h_outPinned = nullptr;
    err = cudaMallocHost(&h_outPinned, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to allocate pinned host memory");
    }

    const int threadsPerBlock = 256;
    const int blocks = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);
    geluKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_in, d_out, N);

    err = cudaMemcpyAsync(h_outPinned, d_out, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_outPinned);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to copy data from device");
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFreeHost(h_outPinned);
        cudaStreamDestroy(stream);
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }

    std::vector<float> result(N);
    std::memcpy(result.data(), h_outPinned, bytes);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_outPinned);
    cudaStreamDestroy(stream);

    return result;
}