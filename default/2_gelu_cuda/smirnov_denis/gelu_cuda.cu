#include <cuda/cmath>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "gelu_cuda.h"

__global__ void vecGelu(float* X, float* Y, size_t n) {
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x;
    if (workIndex < n) {
        float inner = 0.79788456f * X[workIndex] * (1.f + 0.044715f * X[workIndex] * X[workIndex]);
        Y[workIndex] = 0.5f * X[workIndex] * (1.f + tanhf(inner));
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    float* X = nullptr;
    float* Y = nullptr;

    cudaMalloc(&X, n * sizeof(float));
    cudaMalloc(&Y, n * sizeof(float));
    
    cudaMemcpy(X, inptr, n * sizeof(float), cudaMemcpyHostToDevice);
    
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    vecGelu<<<blocks, threads>>>(X, Y, n);
    cudaDeviceSynchronize();

    cudaMemcpy(outptr, Y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(X);
    cudaFree(Y);
    
    return output;
}

#if 0
std::vector<float> GeluRef(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float* inptr = input.data();
    float* outptr = output.data();

    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float y = 0.5f*x*(1 + std::tanh(std::sqrt(2.f/M_PI)*x*(1.f + 0.044715f*x*x)));
        output[i] = y;
    }

    return output;
}

int main() {
    size_t n = 134217728u;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; i++) {
        x[i] = ((float)rand()/RAND_MAX)*20.f - 10.f;
    }

    // Warming-up
    auto y = GeluCUDA(x);

    std::vector<float> yref = GeluRef(x);
    float err = 0.f;
    for (size_t i = 0; i < n; i++) {
        err = std::max(err, std::abs(y[i] - yref[i]));
    }
    printf("max absolute error = %.5g\n", err);
    
    // Performance Measuring
    std::vector<double> time_list;
    for (int i = 0; i < 4; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto y = GeluCUDA(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        time_list.push_back(duration.count());
    }
    double time = *std::min_element(time_list.begin(), time_list.end());
    printf("time = %.2f\n", time);

    return 0;
}
#endif
