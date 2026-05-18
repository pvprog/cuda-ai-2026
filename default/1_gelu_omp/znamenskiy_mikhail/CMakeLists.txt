
project(test_gelu_omp)

cmake_minimum_required(VERSION 3.10)

find_package(OpenMP REQUIRED)

add_executable(test_gelu_omp main.cpp gelu_omp.cpp gelu_omp.h)
target_link_libraries(test_gelu_omp PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(test_gelu_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_gelu_omp PRIVATE -O2)