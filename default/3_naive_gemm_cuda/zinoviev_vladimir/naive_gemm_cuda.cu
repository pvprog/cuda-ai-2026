#include <algorithm>
#include <chrono>
#include <vector>
#include <iostream>
#include <random>
#include <cuda_runtime.h>

#include "naive_gemm_cuda.h"

__global__ void NaiveGemmCUDAKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
        float res = 0.f;
        for(int k = 0; k < n; ++k) {
            res += a[i*n + k] * b[k*n + j];
        }
        c[i*n + j] = 0.f;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const uint size = n * n;
    std::vector<float> c(size);
    const uint block_size = 16;
    const uint num_blocks = (n + block_size - 1) / block_size;
    float *d_a, *d_b, *d_c; 
    cudaMalloc(&d_a, size*sizeof(float));
    cudaMalloc(&d_b, size*sizeof(float));
    cudaMalloc(&d_c, size*sizeof(float));
    cudaMemcpy(d_a, a.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    dim3 blocksPerGrid(num_blocks, num_blocks);
    dim3 threadsPerBlock(block_size, block_size);
    NaiveGemmCUDAKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_c, size*sizeof(float), cudaMemcpyDeviceToHost);
    return c;
}
