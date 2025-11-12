#pragma once
#include <cuda_runtime.h>
#include <cstdio>


inline void checkCuda(cudaError_t status, const char *fn, const char *file, int32_t line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s:%d::%s: %s\n", file, line, fn, cudaGetErrorString(status));
        exit(status);
    }
}
#define CHECK_CUDA(fn) checkCuda((fn), #fn, __FILE__, __LINE__)

inline size_t minMultiple(size_t a, size_t b) {
    return (a / b) * b;
}
#define MIN_MULTIPLE(a, b) minMultiple((a), (b))
