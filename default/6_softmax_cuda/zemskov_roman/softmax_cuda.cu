#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <algorithm>
#include <cassert>

constexpr int WARP_SIZE = 32;
constexpr int VECTOR_SIZE = 4; 

__global__ void softmax_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_count,
    int row_size
) {
    extern __shared__ float shared_mem[];
    float* row_max_shared = shared_mem;
    float* row_sum_shared = &shared_mem[blockDim.x / WARP_SIZE];
    
    int row_idx = blockIdx.x;
    if (row_idx >= row_count) return;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    const float* input_row = input + row_idx * row_size;
    float* output_row = output + row_idx * row_size;
    
    float local_max = -INFINITY;
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        if (i + VECTOR_SIZE <= row_size) {
            float4 vals = reinterpret_cast<const float4*>(input_row + i)[0];
            local_max = fmaxf(local_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
        } else {
            // Handle remaining elements
            for (int j = i; j < min(i + VECTOR_SIZE, row_size); j++) {
                local_max = fmaxf(local_max, input_row[j]);
            }
        }
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lane == 0) {
        row_max_shared[warp_id] = local_max;
    }
    __syncthreads();
    
    if (tid < num_warps) {
        local_max = row_max_shared[tid];
    } else {
        local_max = -INFINITY;
    }
    
    if (num_warps > 1) {
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                local_max = fmaxf(local_max, row_max_shared[tid + offset]);
                row_max_shared[tid] = local_max;
            }
            __syncthreads();
        }
    }
    
    float row_max = row_max_shared[0];
    __syncthreads();
    
    float local_sum = 0.0f;
    
    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        if (i + VECTOR_SIZE <= row_size) {
            float4 vals = reinterpret_cast<const float4*>(input_row + i)[0];
            float4 exp_vals;
            exp_vals.x = __expf(vals.x - row_max);
            exp_vals.y = __expf(vals.y - row_max);
            exp_vals.z = __expf(vals.z - row_max);
            exp_vals.w = __expf(vals.w - row_max);
            reinterpret_cast<float4*>(output_row + i)[0] = exp_vals;
            local_sum += exp_vals.x + exp_vals.y + exp_vals.z + exp_vals.w;
        } else {
            for (int j = i; j < min(i + VECTOR_SIZE, row_size); j++) {
                float exp_val = __expf(input_row[j] - row_max);
                output_row[j] = exp_val;
                local_sum += exp_val;
            }
        }
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (lane == 0) {
        row_sum_shared[warp_id] = local_sum;
    }
    __syncthreads();
    

    if (tid < num_warps) {
        local_sum = row_sum_shared[tid];
    } else {
        local_sum = 0.0f;
    }
    
    if (num_warps > 1) {
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                local_sum += row_sum_shared[tid + offset];
                row_sum_shared[tid] = local_sum;
            }
            __syncthreads();
        }
    }
    
    float row_sum = row_sum_shared[0];
    float inv_sum = __frcp_rn(row_sum);
    __syncthreads();
    

    for (int i = tid * VECTOR_SIZE; i < row_size; i += blockDim.x * VECTOR_SIZE) {
        if (i + VECTOR_SIZE <= row_size) {
            float4 vals = reinterpret_cast<const float4*>(output_row + i)[0];
            vals.x *= inv_sum;
            vals.y *= inv_sum;
            vals.z *= inv_sum;
            vals.w *= inv_sum;
            reinterpret_cast<float4*>(output_row + i)[0] = vals;
        } else {
            for (int j = i; j < min(i + VECTOR_SIZE, row_size); j++) {
                output_row[j] *= inv_sum;
            }
        }
    }
}


__global__ void softmax_kernel_multi_row(
    const float* __restrict__ input,
    float* __restrict__ output,
    int row_count,
    int row_size
) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= row_count) return;
    
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    
    const float* input_row = input + row_idx * row_size;
    float* output_row = output + row_idx * row_size;
    
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];
    
    float local_max = -INFINITY;
    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(local_max, input_row[i]);
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    if (lane == 0) {
        shared_max[threadIdx.y] = local_max;
    }
    __syncthreads();
    
    float row_max = shared_max[threadIdx.y];
    
    float local_sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float exp_val = __expf(input_row[i] - row_max);
        output_row[i] = exp_val;
        local_sum += exp_val;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (lane == 0) {
        shared_sum[threadIdx.y] = local_sum;
    }
    __syncthreads();
    
    float row_sum = shared_sum[threadIdx.y];
    float inv_sum = __frcp_rn(row_sum);
    
    for (int i = tid; i < row_size; i += blockDim.x) {
        output_row[i] *= inv_sum;
    }
}

struct SoftmaxState {
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t allocated_elements = 0;
    
    std::vector<float> result;
    
    ~SoftmaxState() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }
};

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    static SoftmaxState state;
    
    int total_elements = input.size();
    int row_size = total_elements / row_count;
    
    assert(total_elements % row_count == 0);
    assert(row_size > 0);
    
    if (state.allocated_elements != total_elements) {
        if (state.d_input) {
            cudaFree(state.d_input);
            cudaFree(state.d_output);
        }
        
        cudaMalloc(&state.d_input, total_elements * sizeof(float));
        cudaMalloc(&state.d_output, total_elements * sizeof(float));
        
        state.result.resize(total_elements);
        state.allocated_elements = total_elements;
    }
    
    cudaMemcpy(state.d_input, input.data(), total_elements * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    if (row_size <= 256) {
        int rows_per_block = std::min(8, row_count);
        dim3 block_dim(WARP_SIZE, rows_per_block, 1);
        dim3 grid_dim((row_count + rows_per_block - 1) / rows_per_block, 1, 1);
        int shared_mem_size = rows_per_block * 2 * sizeof(float);
        
        softmax_kernel_multi_row<<<grid_dim, block_dim, shared_mem_size>>>(
            state.d_input, state.d_output, row_count, row_size
        );
        
    } else {
        int threads_per_block;
        if (row_size >= 16384) {
            threads_per_block = 256;
        } else if (row_size >= 4096) {
            threads_per_block = 192;
        } else {
            threads_per_block = 128;
        }
        
        dim3 block_dim(threads_per_block, 1, 1);
        dim3 grid_dim(row_count, 1, 1);
        int shared_mem_size = (threads_per_block / WARP_SIZE) * 2 * sizeof(float);
        
        softmax_kernel_optimized<<<grid_dim, block_dim, shared_mem_size>>>(
            state.d_input, state.d_output, row_count, row_size
        );
    }
    
    cudaMemcpy(state.result.data(), state.d_output, total_elements * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    return state.result;
}