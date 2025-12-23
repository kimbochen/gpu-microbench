#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda.h>
#include "utils.h"

// #define CLUSTER_SIZE 8
// #define THREADS_PER_CTA 1024
// #define LOAD_T float4

#define N_ITERS 1000
#define SMEM_SIZE 32768  // Greater than 24 KiB to force 1 CTA per SM

namespace cg = cooperative_groups;
constexpr uint32_t LOAD_SIZE = sizeof(LOAD_T);

// sudo $(which ncu) --clock-control none --metrics sm__sass_inst_executed_op_dshared_ld.sum.per_second ./dsmem_pair

template<typename T>
__device__ __forceinline__ float dsmem_load(const T *remote_buffer);

template<>
__device__ __forceinline__ float dsmem_load<float>(const float *remote_buffer_addr) {
    float dummy_val;
    asm volatile(
        "ld.shared::cluster.f32 %0, [%1];"
        : "=f"(dummy_val)
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return dummy_val;
}

template<>
__device__ __forceinline__ float dsmem_load<float2>(const float2 *remote_buffer_addr) {
    float dummy_val[2];
    asm volatile(
        "ld.shared::cluster.v2.f32 {%0, %1}, [%2];"
        : "=f"(dummy_val[0]), "=f"(dummy_val[1])
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return dummy_val[0] + dummy_val[1];
}

template<>
__device__ __forceinline__ float dsmem_load<float4>(const float4 *remote_buffer_addr) {
    float dummy_val[4];
    asm volatile(
        "ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dummy_val[0]), "=f"(dummy_val[1]), "=f"(dummy_val[2]), "=f"(dummy_val[3])
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return dummy_val[0] + dummy_val[1] + dummy_val[2] + dummy_val[3];
}


__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void distributedSharedMemoryPair(float *data) {
    extern __shared__ LOAD_T buffer[];
    const size_t N_buffer = SMEM_SIZE / LOAD_SIZE;
    cg::cluster_group cluster = cg::this_cluster();

    for (int32_t i = 0; i < N_buffer; i += blockDim.x) {
        buffer[i] = reinterpret_cast<LOAD_T*>(data)[N_buffer * blockIdx.x + i];
    }
    __syncthreads();
    cluster.sync();

    LOAD_T *remote_buffer = cluster.map_shared_rank(buffer, cluster.block_rank() ^ 1);
    float acc = 0.0f;

    for (int32_t j = 0; j < N_ITERS; j++) {
        for (int32_t i = threadIdx.x; i < N_buffer; i += blockDim.x) {
            acc += dsmem_load<LOAD_T>(remote_buffer + i);
        }
    }
    cluster.sync();

    data[cg::this_grid().thread_rank()] = acc;
}


void benchDistributedSharedMemoryPairThroughput() {
    int32_t num_ctas = 144;
    size_t data_size = num_ctas * SMEM_SIZE;
    size_t N = data_size / sizeof(float);
    float *data, *d_data;

    data = (float*) malloc(data_size);
    srand((uint32_t) CLUSTER_SIZE + THREADS_PER_CTA + LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        data[i] = rand();
    }
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = num_ctas;
    config.blockDim = THREADS_PER_CTA;
    config.dynamicSmemBytes = SMEM_SIZE;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, distributedSharedMemoryPair, d_data);

    cudaFree(d_data);
    free(data);
}


int main(int argc, char **argv) {
    benchDistributedSharedMemoryPairThroughput();
    return 0;
}
