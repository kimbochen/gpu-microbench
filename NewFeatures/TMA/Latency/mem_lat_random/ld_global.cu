#include <algorithm>
#include <iomanip>
#include <random>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <cuda/barrier>

#include "dtime.hpp"
#include "gpu-clock.cuh"
#include "gpu-error.h"
#include "MeasurementSeries.hpp"

const int32_t UNROLL_FACTOR = 32;
const int32_t SKIP_FACTOR = 16;


__global__ void pchase(int64_t *buff, int64_t *__restrict__ dummy_buff, int64_t N)
{
    uint32_t tid = threadIdx.x;
    int64_t *src_addr = buff + tid;
    int64_t cur_val;

    asm volatile(
        "ld.global.cs.s64 %0, [%1];"
        : "=l"(cur_val)
        : "l"(src_addr)
        : "memory"
    );
	asm volatile("bar.sync 0;");

    #pragma unroll 1
    for (int64_t n = 0; n < N; n += UNROLL_FACTOR) {
    #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            int64_t *next_addr = (int64_t*) cur_val;
            int64_t next_val;
            asm volatile(
                "ld.global.cs.s64 %0, [%1];"
                : "=l"(next_val)
                : "l"(next_addr)
                : "memory"
            );
            cur_val = next_val;
        }
    }

    if (tid > 12313) {
        dummy_buff[tid] = cur_val;
    }
}


int main(int argc, char **argv)
{
    std::mt19937 rng(3985);
    uint32_t clock = getGPUClock(0);

    for (int64_t LEN = 16; LEN < (1 << 24); LEN = LEN * 1.04 + 32) {
        if (LEN * SKIP_FACTOR * sizeof(int64_t) > 120 * 1048576) {
            LEN = LEN * 3 / 2;
        }
        uint64_t buff_size = LEN * SKIP_FACTOR * sizeof(int64_t);
        const int64_t n_iters = std::max(LEN, (int64_t)10000000);

        std::vector<int64_t> order(LEN);
        for (int64_t i = 0; i < LEN - 1; i++) {
            order[i] = i + 1;
        }
        order[LEN - 1] = 0;
        std::shuffle(begin(order), end(order) - 1, rng);

        int64_t *buff = nullptr;
        int64_t *d_buff = nullptr;
        int64_t *dummy_buff = nullptr;

        GPU_ERROR(cudaMallocManaged(&buff, buff_size));
        GPU_ERROR(cudaMalloc(&d_buff, buff_size));
        GPU_ERROR(cudaMallocManaged(&dummy_buff, sizeof(int64_t)));

        for (int64_t i = 0, idx = 0; i < LEN; i++) {
            int64_t next_idx = SKIP_FACTOR * order[i];
            buff[idx] = (int64_t)(d_buff + next_idx);
            idx = next_idx;
        }
        cudaMemcpy(d_buff, buff, buff_size, cudaMemcpyHostToDevice);
        GPU_ERROR(cudaDeviceSynchronize());

        MeasurementSeries times;
        for (int i = 0; i < 8; i++) {
            double start = dtime();

            pchase<<<1, 1>>>(d_buff, dummy_buff, n_iters);
            GPU_ERROR(cudaDeviceSynchronize());

            double end = dtime();
            times.add(end - start);
            GPU_ERROR(cudaDeviceSynchronize());
        }
        GPU_ERROR(cudaGetLastError());

        double dt = times.minValue();
        std::cout << std::setw(9) << n_iters << " "
            << std::setw(5) << clock << " "
            << std::setw(8) << buff_size / 1024 << " "
            << std::fixed << std::setprecision(1) << std::setw(8) << dt * 1000 << " "
            << std::setw(7) << std::setprecision(1) << dt / n_iters * clock * 1000 * 1000 << std::endl;

        GPU_ERROR(cudaFree(buff));
        GPU_ERROR(cudaFree(d_buff));
        GPU_ERROR(cudaFree(dummy_buff));
    }

    std::cout << '\n';

    return 0;
}
