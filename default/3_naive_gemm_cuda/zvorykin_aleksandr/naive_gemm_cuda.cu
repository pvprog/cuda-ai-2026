#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

// CUDA tile size
static const int s_TileSize = 16;

// Anonymous namespace
namespace
{    
    __global__ void naiveGemmImpl(const float *a, const float *b, float *c, int n)
    {
        __shared__ float sharedA[s_TileSize][s_TileSize];
        __shared__ float sharedB[s_TileSize][s_TileSize];

        int tX = threadIdx.x;
        int tY = threadIdx.y;
        int cRow = blockIdx.y * s_TileSize + tY;
        int cCol = blockIdx.x * s_TileSize + tX;

        if (cRow >= n || cCol >= n) return;

        float sum = 0.0f;
        int numTiles = (n + s_TileSize - 1) / s_TileSize;

        for (int t = 0, tile = 0; t < numTiles; ++t, tile += s_TileSize)
        {
            sharedA[tY][tX] = (tX + tile < n) ? a[cRow * n + tX + tile] : 0.0f;
            sharedB[tY][tX] = (tY + tile < n) ? b[(tY + tile) * n + cCol] : 0.0f;
            __syncthreads();

            for (int k = 0; k < s_TileSize; ++k)
                sum += sharedA[tY][k] * sharedB[k][tX];
            __syncthreads();
        }

        c[cRow * n + cCol] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) 
{
    // Place your implementation here
    const float* aData = a.data();
    const float* bData = b.data();
    const int dataSize = n * n;

    float* deviceA = nullptr;
    cudaMalloc(&deviceA, dataSize * sizeof(float));
    float* deviceB = nullptr;
    cudaMalloc(&deviceB, dataSize * sizeof(float));
    float* deviceC = nullptr;
    cudaMalloc(&deviceC, dataSize * sizeof(float));

    cudaMemcpy(deviceA, aData, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, bData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(s_TileSize, s_TileSize);
    dim3 blockDims((n + s_TileSize - 1) / s_TileSize, (n + s_TileSize - 1) / s_TileSize);
    naiveGemmImpl<<<blockDims, threadsPerBlock>>>(deviceA, deviceB, deviceC, n);

    std::vector<float> c(dataSize);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), deviceC, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return c;
}
