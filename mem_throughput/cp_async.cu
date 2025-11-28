#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

constexpr int32_t THREADS_PER_BLOCK = 1024;

using load_t = float4;  // [float, float2, float4]
constexpr int32_t VECTOR_WIDTH = sizeof(load_t) / sizeof(float);
constexpr int32_t LOAD_SIZE = sizeof(load_t);


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


void benchAsyncCopyThroughput(int32_t num_blks_factor) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_DATA_VOLUME, (num_blks * THREADS_PER_BLOCK * LOAD_SIZE));
    size_t N = arr_size / sizeof(float);

    arr = (float*) malloc(arr_size);
    srand((uint32_t) num_blks);
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand();
    }
    CHECK_CUDA(cudaMalloc(&d_arr, arr_size));
    CHECK_CUDA(cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    asyncCopyKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaPeekAtLastError());

    float t_elapsed;
    cudaEventElapsedTime(&t_elapsed, start, stop);
    printf("%d, %d, %lu, %.5f\n", num_blks_factor, LOAD_SIZE, arr_size, t_elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_arr));
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 3) {
        puts("Usage: ./cp_async [DEVICE_ID] [NUM_BLKS_FACTOR]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchAsyncCopyThroughput(atoi(argv[2]));

    return 0;
}
