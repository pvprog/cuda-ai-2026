#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    if (n <= 0)
        return std::vector<float>();

    const std::size_t elems = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elems * sizeof(float);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    std::vector<float> c(elems);

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(stream);

    return c;
}
