#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>

constexpr float ALPHA = 1.0f;
constexpr float BETA = 0.0f;

struct StatsVars{
    cublasHandle_t handle;
    bool initialized = false;

    size_t stat_size = 0;
    std::vector<float> c;
    
    // device memory
    float * a_dev = nullptr;
    float * b_dev = nullptr;
    float * c_dev = nullptr;

    ~StatsVars(){
        if (a_dev) cudaFree(a_dev);
        if (b_dev) cudaFree(b_dev);
        if (c_dev) cudaFree(c_dev);
        
        if (initialized) {
            cublasDestroy(handle);
        }
    }
};

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    static StatsVars vars;
    
    if (!vars.initialized) {
        cublasCreate(&vars.handle);
        vars.initialized = true;
    }

    const size_t matrix_size = a.size() * sizeof(float);
    
    // Reallocate if size changed
    if (vars.stat_size != matrix_size){
        if (vars.a_dev) {
            cudaFree(vars.a_dev);
            cudaFree(vars.b_dev);
            cudaFree(vars.c_dev);
        }

        cudaMalloc(&vars.a_dev, matrix_size);
        cudaMalloc(&vars.b_dev, matrix_size);
        cudaMalloc(&vars.c_dev, matrix_size);
        
        vars.c.resize(a.size());
        vars.stat_size = matrix_size;
    }
    
    // Synchronous copy to device (no need for pinned memory)
    cudaMemcpy(vars.a_dev, a.data(), matrix_size, 
                          cudaMemcpyHostToDevice);
    cudaMemcpy(vars.b_dev, b.data(), matrix_size, 
                          cudaMemcpyHostToDevice);
    
    cublasSgemm(vars.handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n, n, n,
                             &ALPHA,
                             vars.b_dev, n,
                             vars.a_dev, n,
                             &BETA,
                             vars.c_dev, n);
    
    // Synchronous copy back from device
    cudaMemcpy(vars.c.data(), vars.c_dev, matrix_size,
                          cudaMemcpyDeviceToHost);
   
    return vars.c;
}