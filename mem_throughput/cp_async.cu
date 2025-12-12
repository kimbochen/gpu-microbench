#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "utils.h"

/* Benchmark command
 * sudo $(which ncu) --clock-control none --metrics dram__bytes_read.sum.per_second,sm__cycles_elapsed.avg,sm__sass_inst_executed_op_ldgsts.sum,sm__sass_inst_executed_op_ldgsts.sum.per_cycle_elapsed ./cp_async
 */

using load_t = float4;  // [float, float2, float4]
constexpr int32_t VECTOR_WIDTH = sizeof(load_t) / sizeof(float);
constexpr int32_t LOAD_SIZE = sizeof(load_t);

constexpr int32_t CTAS_PER_SM = 2;
constexpr int32_t NUM_STAGES = 3;
constexpr int32_t THREADS_PER_BLOCK = 512;


__global__ void asyncCopyKernel(float *arr, size_t N) {
    __shared__ load_t buff[NUM_STAGES][THREADS_PER_BLOCK];

    int32_t tid = threadIdx.x;
    int32_t base = blockDim.x * blockIdx.x + threadIdx.x;
    int32_t stride = NUM_STAGES * (gridDim.x * blockDim.x);
    int32_t num_iters = ((N / VECTOR_WIDTH) - base) / stride;


    for (int32_t i = 0; i < num_iters; i++) {
        __pipeline_wait_prior(NUM_STAGES - 1);

        int32_t slot = i % NUM_STAGES;
        int32_t offset = (base + stride * i) * VECTOR_WIDTH;

        __pipeline_memcpy_async(buff[slot] + tid, arr + offset, LOAD_SIZE);  // LDGSTS
        __pipeline_commit();                                                 // LDGDEPBAR
    }

    __pipeline_wait_prior(NUM_STAGES - 1);  // DEPBAR.LE SB, NUM_STAGES-1
}


void benchAsyncCopyThroughput(int32_t blk_factor) {
    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * blk_factor;
    size_t arr_size = minMultiple(MAX_DATA_VOLUME, (num_blks * NUM_STAGES * THREADS_PER_BLOCK * LOAD_SIZE));
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
    benchAsyncCopyThroughput(CTAS_PER_SM);
    return 0;
}
