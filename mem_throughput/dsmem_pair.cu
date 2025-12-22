#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda.h>
#include "utils.h"

#define THREADS_PER_CTA 128
#define LOAD_T float

namespace cg = cooperative_groups;
constexpr uint32_t VECTOR_WIDTH = sizeof(LOAD_T) / sizeof(float);
constexpr uint32_t LOAD_SIZE = sizeof(LOAD_T);

// sudo $(which ncu) --metrics sm__sass_inst_executed_op_dshared_ld.sum.per_second ./dsmem_pair


__global__ __cluster_dims__(2, 1, 1)
void distributedSharedMemory(float *data, size_t N) {
    extern __shared__ LOAD_T buffer[];
    cg::cluster_group cluster = cg::this_cluster();
    float acc = 0.0f;

    for (int32_t i = cg::this_grid().thread_rank(); i < N / VECTOR_WIDTH; i += gridDim.x * blockDim.x) {
        buffer[threadIdx.x] = reinterpret_cast<LOAD_T*>(data)[i];
        cluster.sync();

        float dummy_val;
        LOAD_T *dst_buffer = cluster.map_shared_rank(buffer, cluster.block_rank() ^ 1);
        asm volatile(
            "ld.shared::cluster.f32 %0, [%1];"
            : "=f"(dummy_val)
            : "l"(dst_buffer + threadIdx.x)
            : "memory"
        );
        cluster.sync();

        acc += dummy_val;
    }

    data[cg::this_grid().thread_rank()] = acc;
}


void benchDistributedSharedMemoryPairThroughput() {
    int32_t num_ctas = NUM_SMS * 2;
    size_t data_size = getDataVolume(num_ctas * THREADS_PER_CTA * LOAD_SIZE);
    size_t N = data_size / sizeof(float);
    float *data, *d_data;

    data = (float*) malloc(data_size);
    srand((uint32_t) THREADS_PER_CTA + LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        data[i] = rand();
    }
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = num_ctas;
    config.blockDim = THREADS_PER_CTA;
    config.dynamicSmemBytes = THREADS_PER_CTA * LOAD_SIZE;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, distributedSharedMemory, d_data, N);

    cudaFree(d_data);
    free(data);
}


int main(int argc, char **argv) {
    benchDistributedSharedMemoryPairThroughput();
    return 0;
}
