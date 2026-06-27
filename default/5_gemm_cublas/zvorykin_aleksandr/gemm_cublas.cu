#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n)
{
    // Place your implementation here
    static cublasHandle_t cublas_handle = nullptr;
    static cudaStream_t stream = nullptr;
    static float* deviceA = nullptr;
    static float* deviceB = nullptr;
    static float* deviceC = nullptr;
    static int last_n = 0;
    static bool initialized = false;
  
    if (!initialized)
    {
      cublasCreate(&cublas_handle);
      cudaStreamCreate(&stream);
      cublasSetStream(cublas_handle, stream);
      cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
      initialized = true;
    }
  
    const size_t dataSize = sizeof(float) * static_cast<size_t>(n) * n;
  
    if (n != last_n)
    {
      if (deviceA)
      {
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
      }
      cudaMalloc(&deviceA, dataSize);
      cudaMalloc(&deviceB, dataSize);
      cudaMalloc(&deviceC, dataSize);
      last_n = n;
    }
  
    const float alpha = 1.0f;
    const float beta = 0.0f;
  
    cudaMemcpy(deviceA, a.data(), dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), dataSize, cudaMemcpyHostToDevice);
  
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                &alpha, deviceA, n, deviceB, n, &beta, deviceC, n);
  
    std::vector<float> result(static_cast<size_t>(n) * n);
    cudaMemcpy(result.data(), deviceC, dataSize, cudaMemcpyDeviceToHost);
  
    return result;
}
