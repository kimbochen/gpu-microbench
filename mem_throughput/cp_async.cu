#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

using load_t = float4;  // [float, float2, float4]
constexpr int32_t VECTOR_WIDTH = sizeof(load_t) / sizeof(float);
constexpr int32_t LOAD_SIZE = sizeof(load_t);
constexpr int32_t THREADS_PER_BLOCK = 1024;


__global__ void asyncCopyKernel(float *arr, size_t N) {
    __shared__ load_t buff[THREADS_PER_BLOCK];

    size_t tid = threadIdx.x;
    size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = offset; i < N / VECTOR_WIDTH; i += stride) {
        __pipeline_memcpy_async(buff + tid, arr + i * VECTOR_WIDTH, LOAD_SIZE);  // LDGSTS
        __pipeline_commit();                                                     // LDGDEPBAR
        __pipeline_wait_prior(0);                                                // DEPBAR.LE SB, 0
    }
}


void benchAsyncCopyThroughput(int32_t blk_factor) {
    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * blk_factor;
    size_t arr_size = minMultiple(MAX_DATA_VOLUME, (num_blks * THREADS_PER_BLOCK * LOAD_SIZE));
    size_t N = arr_size / sizeof(float);

    arr = (float*) malloc(arr_size);
    srand((uint32_t) num_blks + LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand();
    }
    cudaMalloc(&d_arr, arr_size);
    cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float t_elapsed;
    DO_BENCH(t_elapsed, asyncCopyKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, N));
    printf("blk_factor=%d, load_size=%d, arr_size=%lu, t_elapsed=%.5f\n", blk_factor, LOAD_SIZE, arr_size, t_elapsed);

    cudaFree(d_arr);
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        puts("Usage: ./cp_async [BLK_FACTOR]");
        return 1;
    }

    benchAsyncCopyThroughput(atoi(argv[1]));

    return 0;
}
