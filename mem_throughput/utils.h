#ifndef UTILS_H
#define UTILS_H

constexpr int32_t NUM_SMS = 148;                              // B200 has 148 SMs
constexpr int32_t L2_SIZE = 132644864;                        // B200 L2 cache is 126.5 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

inline size_t minMultiple(size_t a, size_t b) {
    return (a / b) * b;
}

#define DO_BENCH(t_elapsed, ...)                       \
    do {                                               \
        void *flush_arr;                               \
        cudaMalloc(&flush_arr, L2_SIZE);               \
        cudaMemset(flush_arr, 0xA5, L2_SIZE);          \
        cudaDeviceSynchronize();                       \
        cudaFree(flush_arr);                           \
        cudaEvent_t start, stop;                       \
        cudaEventCreate(&start);                       \
        cudaEventCreate(&stop);                        \
        cudaEventRecord(start);                        \
        __VA_ARGS__;                                   \
        cudaDeviceSynchronize();                       \
        cudaEventRecord(stop);                         \
        cudaEventSynchronize(stop);                    \
        cudaPeekAtLastError();                         \
        cudaEventElapsedTime(&t_elapsed, start, stop); \
        cudaEventDestroy(start);                       \
        cudaEventDestroy(stop);                        \
    } while (0)

#endif
