#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

constexpr int32_t NUM_SMS = 148;                            // B200 has 148 SMs
constexpr int32_t L2_SIZE = 132644864;                      // B200 L2 cache is 126.5 MiB
constexpr int32_t THREADS_PER_BLOCK = 1024;
constexpr size_t MAX_LOAD_SIZE = 2LL * 1024 * 1024 * 1024;  // 2 GB


__global__ void asyncCopyKernel(float *arr, int32_t load_size, size_t N) {
    __shared__ float buff[(THREADS_PER_BLOCK * 16) / sizeof(float)];

    size_t tid = threadIdx.x * (load_size / sizeof(float));
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * (load_size / sizeof(float));
    size_t stride = (gridDim.x * blockDim.x) * (load_size / sizeof(float));

    for (size_t i = offset; i < N; i += stride) {
        __pipeline_memcpy_async(buff + tid, arr + i, load_size);  // LDGSTS
        __pipeline_commit();                                      // LDGDEPBAR
        __pipeline_wait_prior(0);                                 // DEPBAR.LE SB, 0
    }
}


void benchAsyncCopyThroughput(int32_t num_blks_factor, int32_t load_size) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_LOAD_SIZE, (num_blks * THREADS_PER_BLOCK * load_size));
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
    asyncCopyKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, load_size, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaPeekAtLastError());

    float t_elapsed;
    cudaEventElapsedTime(&t_elapsed, start, stop);
    printf("%d, %d, %lu, %.5f\n", num_blks_factor, load_size, arr_size, t_elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_arr));
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        puts("Usage: ./cp_async [DEVICE_ID] [NUM_BLKS_FACTOR] [LOAD_SIZE]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchAsyncCopyThroughput(atoi(argv[2]), atoi(argv[3]));

    return 0;
}
