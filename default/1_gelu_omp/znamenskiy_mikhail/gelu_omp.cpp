#include "gelu_omp.h"

#include <cmath>
#include <omp.h>
#include <cstring>
#include <iostream>

namespace {

inline double fastTanh(const double x) {
    if (x >= 4.97) return 1.0;
    if (x <= -4.97) return -1.0;
    double x2 = x * x;
    double a = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    double b = 135155.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    return a / b;
}

}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result;
    result.reserve(input.size());

    constexpr float sqr2_mpi = std::sqrt(2.f/M_PI);
    constexpr size_t BLOCK_SIZE = 8;
    const size_t inputSize = input.size();
    const size_t vectorizedSize = inputSize - inputSize % BLOCK_SIZE;

    #pragma omp parallel for
    for (size_t index = 0; index < vectorizedSize; index += BLOCK_SIZE) {
        result[index] = input[index] * input[index] * input[index];
        result[index+1] = input[index+1] * input[index+1] * input[index+1];
        result[index+2] = input[index+2] * input[index+2] * input[index+2];
        result[index+3] = input[index+3] * input[index+3] * input[index+3];
        result[index+4] = input[index+4] * input[index+4] * input[index+4];
        result[index+5] = input[index+5] * input[index+5] * input[index+5];
        result[index+6] = input[index+6] * input[index+6] * input[index+6];
        result[index+7] = input[index+7] * input[index+7] * input[index+7];
        result[index] *= 0.044715f;
        result[index+1] *= 0.044715f;
        result[index+2] *= 0.044715f;
        result[index+3] *= 0.044715f;
        result[index+4] *= 0.044715f;
        result[index+5] *= 0.044715f;
        result[index+6] *= 0.044715f;
        result[index+7] *= 0.044715f;
        result[index] += input[index];
        result[index+1] += input[index+1];
        result[index+2] += input[index+2];
        result[index+3] += input[index+3];
        result[index+4] += input[index+4];
        result[index+5] += input[index+5];
        result[index+6] += input[index+6];
        result[index+7] += input[index+7];
        result[index] *= sqr2_mpi;
        result[index+1] *= sqr2_mpi;
        result[index+2] *= sqr2_mpi;
        result[index+3] *= sqr2_mpi;
        result[index+4] *= sqr2_mpi;
        result[index+5] *= sqr2_mpi;
        result[index+6] *= sqr2_mpi;
        result[index+7] *= sqr2_mpi;
        result[index] = fastTanh(result[index]);
        result[index+1] = fastTanh(result[index+1]);
        result[index+2] = fastTanh(result[index+2]);
        result[index+3] = fastTanh(result[index+3]);
        result[index+4] = fastTanh(result[index+4]);
        result[index+5] = fastTanh(result[index+5]);
        result[index+6] = fastTanh(result[index+6]);
        result[index+7] = fastTanh(result[index+7]);
        result[index] += 1.f;
        result[index+1] += 1.f;
        result[index+2] += 1.f;
        result[index+3] += 1.f;
        result[index+4] += 1.f;
        result[index+5] += 1.f;
        result[index+6] += 1.f;
        result[index+7] += 1.f;
        result[index] *= 0.5f * input[index];
        result[index+1] *= 0.5f * input[index+1];
        result[index+2] *= 0.5f * input[index+2];
        result[index+3] *= 0.5f * input[index+3];
        result[index+4] *= 0.5f * input[index+4];
        result[index+5] *= 0.5f * input[index+5];
        result[index+6] *= 0.5f * input[index+6];
        result[index+7] *= 0.5f * input[index+7];
    }

    #pragma omp parallel for
    for (size_t index = vectorizedSize; index < inputSize; ++index) {
        result[index] = input[index] * input[index] * input[index];
        result[index] *= 0.044715f;
        result[index] += input[index];
        result[index] *= sqr2_mpi;
        result[index] = fastTanh(result[index]);
        result[index] += 1.f;
        result[index] *= 0.5f * input[index];
    }
    return result;
}