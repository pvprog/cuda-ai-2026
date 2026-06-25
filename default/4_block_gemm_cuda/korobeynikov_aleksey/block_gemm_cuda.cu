#include "block_gemm_cuda.h"


static constexpr int BLOCK_SIZE = 16;

__global__ void gemm_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int block_i = threadIdx.y;
    int block_j = threadIdx.x;

    __shared__ float block_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float block_b[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.f;
    for (int block_n = 0; block_n < gridDim.x; ++block_n) {
        int a_col = block_n * BLOCK_SIZE + block_j;
        int b_row = block_n * BLOCK_SIZE + block_i;

        block_a[block_i * BLOCK_SIZE + block_j] = a[i * n + a_col];
        block_b[block_i * BLOCK_SIZE + block_j] = b[b_row * n + j];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += block_a[block_i * BLOCK_SIZE + k] * block_b[k * BLOCK_SIZE + block_j];
        }
        __syncthreads();
    }

    c[i * n + j] = sum;
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    std::vector<float> result(a.size());

    float* a_gpu = nullptr;
    float* b_gpu = nullptr;
    float* result_gpu = nullptr;

    size_t size_in_bytes = a.size() * sizeof(float);

    cudaMalloc(&a_gpu, size_in_bytes);
    cudaMalloc(&b_gpu, size_in_bytes);
    cudaMalloc(&result_gpu, size_in_bytes);

    cudaMemcpy(a_gpu, a.data(), size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b.data(), size_in_bytes, cudaMemcpyHostToDevice);

    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (n + threads_per_block.x - 1) / threads_per_block.x,
        (n + threads_per_block.y - 1) / threads_per_block.y
    );

    gemm_kernel<<<blocks, threads_per_block>>>(a_gpu, b_gpu, result_gpu, n);

    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), result_gpu, size_in_bytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    return result;
}
