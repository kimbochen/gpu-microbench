#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/ptx>
#include "utils.h"

#define CTAS_PER_SM 2
// #define SMEM_WIDTH 128
// #define SMEM_HEIGHT 2

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;


__global__ void bulkAsyncCopyTensor2D(const __grid_constant__ CUtensorMap tensor_map, uint64_t gmem_height) {
    __shared__ alignas(128) float smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];
    __shared__ alignas(8) uint64_t bar;

    if (threadIdx.x == 0) {
        ptx::mbarrier_init(&bar, blockDim.x);
    }
    __syncthreads();

    uint32_t parity = 0;
    for (int32_t i = SMEM_HEIGHT * blockIdx.x; i < gmem_height; i += SMEM_HEIGHT * gridDim.x) {
        if (threadIdx.x == 0) {
            cg::invoke_one(cg::coalesced_threads(), [&] () {
                int32_t tensor_coords[2];
                tensor_coords[0] = 0;
                tensor_coords[1] = i;

                ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &smem_buffer, &tensor_map, tensor_coords, &bar);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buffer));
            });
        }
        while (!ptx::mbarrier_try_wait_parity(&bar, parity));
        parity ^= 1;
    }
}

void benchBulkAsyncCopyTensor2DThroughput(int32_t ctas_per_sm) {
    int32_t num_ctas = NUM_SMS * ctas_per_sm;
    size_t data_size = minMultiple(MAX_DATA_VOLUME, SMEM_HEIGHT * num_ctas * sizeof(float));

    constexpr uint64_t GMEM_WIDTH = SMEM_WIDTH;
    uint64_t GMEM_HEIGHT = (data_size / sizeof(float)) / GMEM_WIDTH;
    float *data, *d_data;

    data = (float*) malloc(data_size);
    srand((uint32_t) SMEM_WIDTH + SMEM_HEIGHT);
    for (size_t i = 0; i < GMEM_WIDTH * GMEM_HEIGHT; i++) {
        data[i] = rand();
    }
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(float)};
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};
    void *tensor_ptr = d_data;

    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr; 
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled;

    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
    assert(driver_status == cudaDriverEntryPointSuccess);
    cuTensorMapEncodeTiled = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);

    CUresult res = cuTensorMapEncodeTiled(
		&tensor_map,                // CUtensorMap *tensorMap,
		CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
		rank,                       // cuuint32_t tensorRank,
		tensor_ptr,                 // void *globalAddress,
		size,                       // const cuuint64_t *globalDim,
		stride,                     // const cuuint64_t *globalStrides,
		box_size,                   // const cuuint32_t *boxDim,
		elem_stride,                // const cuuint32_t *elementStrides,
		CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
		CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
		CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
		CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
	);

    float t_elapsed;
    DO_BENCH(t_elapsed, bulkAsyncCopyTensor2D<<<num_ctas, 1>>>(tensor_map, GMEM_HEIGHT));
    printf("ctas_per_sm=%d, GMEM=(%lux%lu), SMEM=(%dx%d), t_elapsed=%.5f\n", ctas_per_sm, GMEM_WIDTH, GMEM_HEIGHT, SMEM_WIDTH, SMEM_HEIGHT, t_elapsed);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    free(data);
}


int main(int argc, char **argv) {
    benchBulkAsyncCopyTensor2DThroughput(CTAS_PER_SM);
    return 0;
}
