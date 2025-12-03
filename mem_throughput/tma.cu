#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include "utils.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

constexpr size_t LOAD_SIZE = 1024;
constexpr size_t ELEMS_PER_LOAD = LOAD_SIZE / sizeof(float);
constexpr size_t NUM_LOADS = 4;


__global__ void BulkAsyncCopyKernel(float *arr, size_t N) {
    __shared__ alignas(16) float buff[(LOAD_SIZE * NUM_LOADS) / sizeof(float)];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[NUM_LOADS];

    if (threadIdx.x == 0) {
        for (int32_t i = 0; i < NUM_LOADS; i++) {
            init(&bar[i], 1);
        }
    }
    __syncthreads();

    size_t offset = blockIdx.x;
    size_t stride = gridDim.x;

    for (size_t i = offset, j = 0; i < N / ELEMS_PER_LOAD; i += stride, j++) {
        if (threadIdx.x == 0) {
            cg::invoke_one(cg::coalesced_threads(), [&] {
                float *smem_ptr = buff + ELEMS_PER_LOAD * j;
                float *gmem_ptr = arr + ELEMS_PER_LOAD * i;

                ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, smem_ptr, gmem_ptr, LOAD_SIZE, &bar[j % NUM_LOADS]);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[j % NUM_LOADS], LOAD_SIZE);
            });
        }

        while (!ptx::mbarrier_try_wait_parity(&bar[j % NUM_LOADS], 0));
    }
}


void benchBulkAsyncCopyThroughput(int32_t num_blks_factor) {
    void *flush_arr;
    CHECK_CUDA(cudaMalloc(&flush_arr, L2_SIZE));
    CHECK_CUDA(cudaMemset(flush_arr, 0xA5, L2_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(flush_arr);

    float *arr, *d_arr;
    int32_t num_blks = NUM_SMS * num_blks_factor;
    size_t arr_size = MIN_MULTIPLE(MAX_DATA_VOLUME, (num_blks * LOAD_SIZE));
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
    BulkAsyncCopyKernel<<<num_blks, 1>>>(d_arr, N);
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
        puts("Usage: ./tma [DEVICE_ID] [NUM_BLKS_FACTOR]");
        return 1;
    }

    cudaSetDevice(atoi(argv[1]));
    benchBulkAsyncCopyThroughput(atoi(argv[2]));

    return 0;
}
