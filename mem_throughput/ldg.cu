#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

constexpr int32_t THREADS_PER_BLOCK = 1024;
constexpr int32_t UNROLL_FACTOR = 2;

using load_t = float4;  // [float, float2, float4]
constexpr int32_t VECTOR_WIDTH = sizeof(load_t) / sizeof(float);
constexpr int32_t LOAD_SIZE = sizeof(load_t);


__global__ void LDGSTGKernel(float *src, float *dst, size_t N) {
    load_t buff[UNROLL_FACTOR];

    size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = offset; i < N / VECTOR_WIDTH; i += stride * UNROLL_FACTOR) {
        #pragma unroll
        for (int32_t j = 0; j < UNROLL_FACTOR; j++) {
            buff[j] = reinterpret_cast<load_t*>(src)[i + stride * j];
        }

        #pragma unroll
        for (int32_t j = 0; j < UNROLL_FACTOR; j++) {
            reinterpret_cast<load_t*>(dst)[i + stride * j] = buff[j];
        }
    }
}


void benchLDGSTGThroughput(int32_t num_blks_factor) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *src, *d_src;
    float *d_dst;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_DATA_VOLUME, (num_blks * THREADS_PER_BLOCK * LOAD_SIZE * UNROLL_FACTOR));
    size_t N = arr_size / sizeof(float);

    src = (float*) malloc(arr_size);
    srand((uint32_t) num_blks);
    for (size_t i = 0; i < N; i++) {
        src[i] = rand();
    }
    CHECK_CUDA(cudaMalloc(&d_src, arr_size));
    CHECK_CUDA(cudaMalloc(&d_dst, arr_size));
    CHECK_CUDA(cudaMemcpy(d_src, src, arr_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    LDGSTGKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_src, d_dst, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaPeekAtLastError());

    float t_elapsed;
    cudaEventElapsedTime(&t_elapsed, start, stop);
    printf("%d, %d, %lu, %.5f\n", num_blks_factor, LOAD_SIZE, arr_size * 2, t_elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    free(src);
}


int main(int argc, char **argv) {
    if (argc != 3) {
        puts("Usage: ./ldg [DEVICE_ID] [NUM_BLKS_FACTOR]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchLDGSTGThroughput(atoi(argv[2]));

    return 0;
}
