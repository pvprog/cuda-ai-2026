#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

// Anonymous namespace
namespace
{
    // CUDA constants
    constexpr int s_TileSize = 32;
    constexpr int s_VectorWidth = 4;
    
    __global__ void OptimizedGemm(const float *a, const float *b, float *c, int n) 
    {
        __shared__ float tileA[s_TileSize][s_TileSize];
        __shared__ float tileB[s_TileSize][s_TileSize];
        
        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;
        
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        
        int row = by * s_TileSize + ty;
        int col = (bx * s_TileSize + tx) * s_VectorWidth;
        
        if (row >= n) return;
        
        for (int t = 0, tSize = 0; t < (n + s_TileSize - 1) / s_TileSize; ++t, tSize += s_TileSize)
        {
            if (tSize + tx < n)
                tileA[ty][tx] = a[row * n + tSize + tx];
            else
                tileA[ty][tx] = 0.0f;
            
            int b_row = tSize + ty;
            if (b_row < n && col + 3 < n && ((uintptr_t)(&b[b_row * n + col]) & 0xF) == 0)
            {
                float4 b_vec = reinterpret_cast<const float4*>(&b[b_row * n + col])[0];
                tileB[ty][tx * 4 + 0] = b_vec.x;
                tileB[ty][tx * 4 + 1] = b_vec.y;
                tileB[ty][tx * 4 + 2] = b_vec.z;
                tileB[ty][tx * 4 + 3] = b_vec.w;
            }
            else if (b_row < n) 
            {
                for (int v = 0; v < s_VectorWidth && col + v < n; ++v)
                    tileB[ty][tx * 4 + v] = b[b_row * n + col + v];
            }
            else 
            {
                for (int v = 0; v < s_VectorWidth; ++v)
                    tileB[ty][tx * 4 + v] = 0.0f;
            }
            
            __syncthreads();
            
            #pragma unroll
            for (int k = 0; k < s_TileSize; ++k)
            {
                float a_val = tileA[ty][k];
                sum0 += a_val * tileB[k][tx * 4 + 0];
                sum1 += a_val * tileB[k][tx * 4 + 1];
                sum2 += a_val * tileB[k][tx * 4 + 2];
                sum3 += a_val * tileB[k][tx * 4 + 3];
            }
            
            __syncthreads();
        }
        
        if (col < n) c[row * n + col] = sum0;
        if (col + 1 < n) c[row * n + col + 1] = sum1;
        if (col + 2 < n) c[row * n + col + 2] = sum2;
        if (col + 3 < n) c[row * n + col + 3] = sum3;
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

    float* devA = nullptr;
    cudaMalloc(&devA, dataSize * sizeof(float));
    float* devB = nullptr;
    cudaMalloc(&devB, dataSize * sizeof(float));
    float* devC = nullptr;
    cudaMalloc(&devC, dataSize * sizeof(float));

    cudaMemcpy(devA, aData, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, bData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 blockCount((n / s_VectorWidth + s_TileSize - 1) / s_TileSize, 
                    (n + s_TileSize - 1) / s_TileSize);
    
    OptimizedGemm<<<blockCount, threadsPerBlock>>>(devA, devB, devC, n);
    cudaDeviceSynchronize();

    std::vector<float> c(dataSize);
    cudaMemcpy(c.data(), devC, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return c;
}
