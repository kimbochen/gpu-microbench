#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include "utils.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;

constexpr int32_t NUM_SMS = 148;                            // B200 has 148 SMs
constexpr int32_t L2_SIZE = 132644864;                      // B200 L2 cache is 126.5 MiB
constexpr size_t MAX_LOAD_SIZE = 2LL * 1024 * 1024 * 1024;  // 2 GB


__global__ void BulkAsyncCopyKernel(float *arr, size_t load_size, size_t N) {
    __shared__ alignas(16) float buff[32768 / sizeof(float)];  // 32 KiB of shared memory

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    size_t offset = blockIdx.x * (load_size / sizeof(float));
    size_t stride = gridDim.x * (load_size / sizeof(float));

    for (size_t i = offset; i < N; i += stride) {
        if (threadIdx.x == 0) {
            cuda::memcpy_async(buff, arr + i, cuda::aligned_size_t<16>(load_size), bar);
        }
        barrier::arrival_token token = bar.arrive();
        bar.wait(std::move(token));
    }
}


void benchBulkAsyncCopyThroughput(int32_t num_blks_factor, size_t load_size) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_LOAD_SIZE, (num_blks * load_size));
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
    BulkAsyncCopyKernel<<<num_blks, 1>>>(d_arr, load_size, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaPeekAtLastError());

    float t_elapsed;
    cudaEventElapsedTime(&t_elapsed, start, stop);
    printf("%d, %lu, %lu, %.5f\n", num_blks_factor, load_size, arr_size, t_elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CHECK_CUDA(cudaFree(d_arr));
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        puts("Usage: ./tma [DEVICE_ID] [NUM_BLKS_FACTOR] [LOAD_SIZE]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchBulkAsyncCopyThroughput(atoi(argv[2]), atoi(argv[3]));

    return 0;
}
