#pragma once
#include <cuda_runtime.h>
#include <cstdio>

constexpr int32_t NUM_SMS = 132;                              // H200 has 132 SMs
constexpr int32_t L2_SIZE = 62914560;                         // H200 L2 cache is 60 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

inline size_t minMultiple(size_t a, size_t b) {
    return (a / b) * b;
}
#define MIN_MULTIPLE(a, b) minMultiple((a), (b))

inline void checkCuda(cudaError_t status, const char *fn, const char *file, int32_t line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s:%d::%s: %s\n", file, line, fn, cudaGetErrorString(status));
        exit(status);
    }
}
#define CHECK_CUDA(fn) checkCuda((fn), #fn, __FILE__, __LINE__)
