#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <iostream>

__global__ void vecGemmKernel(const float* A, const float* B, float* C, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int colBlock = blockIdx.x * blockDim.x + threadIdx.x;
    const int colStart = colBlock * 4;

    if (row < n && colStart < n) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int k = 0; k < n; ++k) {
            const float aVal = A[row * n + k];
            const float4 bVec = reinterpret_cast<const float4*>(&B[k * n + colStart])[0];

            acc.x += aVal * bVec.x;
            acc.y += aVal * bVec.y;
            acc.z += aVal * bVec.z;
            acc.w += aVal * bVec.w;
        }

        if (colStart + 3 < n) {
            float4* cVec = reinterpret_cast<float4*>(&C[row * n + colStart]);
            cVec[0] = acc;
        } else {
            if (colStart < n) C[row * n + colStart] = acc.x;
            if (colStart + 1 < n) C[row * n + colStart + 1] = acc.y;
            if (colStart + 2 < n) C[row * n + colStart + 2] = acc.z;
            if (colStart + 3 < n) C[row * n + colStart + 3] = acc.w;
        }
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t matrixSize = static_cast<size_t>(n) * n;
    const size_t byteSize = matrixSize * sizeof(float);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    cudaMalloc(&deviceA, byteSize);
    cudaMalloc(&deviceB, byteSize);
    cudaMalloc(&deviceC, byteSize);

    cudaMemcpy(deviceA, a.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.data(), byteSize, cudaMemcpyHostToDevice);

    const int nColBlocks = (n + 3) / 4;
    const dim3 threadsPerBlock(32, 24);
    const dim3 blocksPerGrid((nColBlocks + threadsPerBlock.x - 1) / threadsPerBlock.x,
                              (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    vecGemmKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, n);

    std::vector<float> c(matrixSize);
    cudaMemcpy(c.data(), deviceC, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return c;
}
