cmake_minimum_required(VERSION 3.20)
project(naive_gemm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(naive_gemm_cuda
    main.cpp
    naive_gemm_cuda.cu
)

target_include_directories(naive_gemm_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
