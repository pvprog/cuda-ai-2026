#include "naive_gemm_cuda.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

std::vector<float> cpuGemm(const std::vector<float>& A, const std::vector<float>& B, int n) {
    std::vector<float> C(n * n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aVal = A[i * n + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += aVal * B[k * n + j];
            }
        }
    }

    return C;
}

void validateResults(const std::vector<float>& cudaResult, const std::vector<float>& cpuResult, int n) {
    float maxAbsError = 0.0f;
    for (size_t i = 0; i < cudaResult.size(); ++i) {
        float absError = std::abs(cudaResult[i] - cpuResult[i]);
        maxAbsError = std::max(maxAbsError, absError);
    }

    std::cout << "Validation: Max absolute error = " << maxAbsError << std::endl;

    if (maxAbsError < 1e-3f) {
        std::cout << "PASS: Accuracy within tolerance" << std::endl;
    } else {
        std::cout << "FAIL: Accuracy below tolerance" << std::endl;
    }
}

std::vector<float> generateRandomMatrix(int n, float minVal = -1.0f, float maxVal = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(minVal, maxVal);

    std::vector<float> matrix(n * n);
    for (auto& elem : matrix) {
        elem = dist(gen);
    }

    return matrix;
}

int main() {
    const int n = 4096;
    const int iterations = 4;

    std::cout << "CUDA Naive GEMM Benchmark" << std::endl;
    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Total elements: " << (n * n) << std::endl;
    std::cout << std::endl;

    std::vector<float> A = generateRandomMatrix(n, -1.0f, 1.0f);
    std::vector<float> B = generateRandomMatrix(n, -1.0f, 1.0f);

    std::cout << "Running warmup..." << std::endl;
    auto C = NaiveGemmCUDA(A, B, n);

    std::cout << "Validating correctness..." << std::endl;
    std::vector<float> CRef(n * n);
    const int smallN = 512;
    std::vector<float> ASmall(smallN * smallN), BSmall(smallN * smallN);
    for (int i = 0; i < smallN; ++i) {
        for (int j = 0; j < smallN; ++j) {
            ASmall[i * smallN + j] = A[i * n + j];
            BSmall[i * smallN + j] = B[i * n + j];
        }
    }
    auto CSmall = NaiveGemmCUDA(ASmall, BSmall, smallN);
    auto CSmallRef = cpuGemm(ASmall, BSmall, smallN);
    validateResults(CSmall, CSmallRef, smallN);

    std::cout << "\nPerformance benchmark (" << iterations << " iterations):" << std::endl;
    std::vector<double> timings;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        C = NaiveGemmCUDA(A, B, n);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        timings.push_back(duration.count());
        std::cout << "  Iteration " << (i + 1) << ": " << duration.count() << "s" << std::endl;
    }

    double minTime = *std::min_element(timings.begin(), timings.end());
    double avgTime = 0.0;
    for (auto t : timings) {
        avgTime += t;
    }
    avgTime /= timings.size();

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Min time: " << minTime << "s" << std::endl;
    std::cout << "  Avg time: " << avgTime << "s" << std::endl;


    return 0;
}
