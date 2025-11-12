#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

constexpr int32_t NUM_SMS = 148;                            // B200 has 148 SMs
constexpr int32_t L2_SIZE = 132644864;                      // B200 L2 cache is 126.5 MiB
constexpr int32_t THREADS_PER_BLOCK = 1024;
constexpr size_t MAX_LOAD_SIZE = 2LL * 1024 * 1024 * 1024;  // 2 GB
constexpr int32_t LOAD_SIZE = 4;                            // Set to [4, 8, 16] to match cp.async
constexpr int32_t ELEMS_PER_LOAD = LOAD_SIZE / sizeof(float);


__global__ void LDGKernel(float *arr, size_t N) {
    __shared__ float buff[(THREADS_PER_BLOCK * 16) / sizeof(float)];

    size_t tid = threadIdx.x * ELEMS_PER_LOAD;
    size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * ELEMS_PER_LOAD;
    size_t stride = (gridDim.x * blockDim.x) * ELEMS_PER_LOAD;

    for (size_t i = offset; i < N; i += stride) {
        #pragma unroll
        for (int32_t j = 0; j < ELEMS_PER_LOAD; j++) {
            buff[tid + j] = arr[i + j];
        }
        #pragma unroll
        for (int32_t j = 0; j < ELEMS_PER_LOAD; j++) {
            arr[i + j] = buff[tid + j] + 1;
        }
    }
}


void benchLDGThroughput(int32_t num_blks_factor) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_LOAD_SIZE, (num_blks * THREADS_PER_BLOCK * LOAD_SIZE));
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
    LDGKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaPeekAtLastError());

    float t_elapsed;
    cudaEventElapsedTime(&t_elapsed, start, stop);
    printf("%d, %d, %lu, %.5f\n", num_blks_factor, LOAD_SIZE, arr_size * 2, t_elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_arr));
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 3) {
        puts("Usage: ./ldg [DEVICE_ID] [NUM_BLKS_FACTOR]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchLDGThroughput(atoi(argv[2]));

    return 0;
}
