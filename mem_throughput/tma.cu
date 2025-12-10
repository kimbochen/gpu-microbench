#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include "utils.h"

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

constexpr int32_t NUM_STAGES = 4;
constexpr int32_t LOAD_SIZE = 12 * 1024 - NUM_STAGES * sizeof(uint64_t);
constexpr size_t ELEMS_PER_LOAD = LOAD_SIZE / sizeof(float);


__global__ void BulkAsyncCopyKernel(float *arr, size_t N) {
    __shared__ alignas(16) float buff[NUM_STAGES][ELEMS_PER_LOAD];
    __shared__ alignas(8) uint64_t bar[NUM_STAGES];

    if (threadIdx.x == 0) {
        for (int32_t s = 0; s < NUM_STAGES; s++) {
            ptx::mbarrier_init(&bar[s], 1);
        }
    }
    __syncthreads();

    int32_t base = ELEMS_PER_LOAD * blockIdx.x;
    int32_t stride = ELEMS_PER_LOAD * gridDim.x;
    int32_t num_iters = (N - base) / stride;

    for (int32_t i = 0; i < num_iters; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;

        while (i >= NUM_STAGES && !ptx::mbarrier_try_wait_parity(&bar[slot], parity));

        int32_t offset = base + i * stride;
        if (threadIdx.x == 0) {
            cg::invoke_one(cg::coalesced_threads(), [&] () {
                ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buff[slot], arr + offset, static_cast<uint32_t>(LOAD_SIZE), &bar[slot]);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[slot], static_cast<uint32_t>(LOAD_SIZE));
            });
        }
    }

    for (int32_t i = num_iters; i < num_iters + NUM_STAGES; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;
        while (!ptx::mbarrier_try_wait_parity(&bar[slot], parity));
    }
}


void benchBulkAsyncCopyThroughput(int32_t blk_factor) {
    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * blk_factor;
    size_t arr_size = minMultiple(MAX_DATA_VOLUME, (num_blks * LOAD_SIZE * NUM_STAGES));
    size_t N = arr_size / sizeof(float);

    arr = (float*) malloc(arr_size);
    srand((uint32_t) NUM_STAGES * LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand();
    }
    cudaMalloc(&d_arr, arr_size);
    cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float t_elapsed;
    DO_BENCH(t_elapsed, BulkAsyncCopyKernel<<<num_blks, 1>>>(d_arr, N));
    printf("blk_factor=%d, load_size=%d, arr_size=%lu, t_elapsed=%.5f\n", blk_factor, LOAD_SIZE, arr_size, t_elapsed);

    cudaFree(d_arr);
    free(arr);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        puts("Usage: ./tma [BLK_FACTOR]");
        return 1;
    }

    benchBulkAsyncCopyThroughput(1);

    return 0;
}
