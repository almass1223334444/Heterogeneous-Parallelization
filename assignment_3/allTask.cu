#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define N 1000000

// =======================
// ШАГ 1 — исправленный CHECK
// =======================
#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// ======================================================
// ЗАДАНИЕ 1
// ======================================================

// 1.1 Только глобальная память
__global__ void mul_global(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

// 1.2 Разделяемая память
__global__ void mul_shared(float* data, float factor, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < n)
        sdata[tid] = data[idx];
    __syncthreads();

    if (idx < n) {
        sdata[tid] *= factor;
        data[idx] = sdata[tid];
    }
}

// ======================================================
// ЗАДАНИЕ 2
// ======================================================

__global__ void vec_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// ======================================================
// ЗАДАНИЕ 3
// ======================================================

// Коалесцированный доступ
__global__ void coalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] *= 2.0f;
}

// Некоалесцированный, но БЕЗОПАСНЫЙ доступ
__global__ void uncoalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 32;
    int new_idx = idx * stride;

    if (new_idx < n) {
        data[new_idx] *= 2.0f;
    }
}

// ======================================================
// ИЗМЕРЕНИЕ ВРЕМЕНИ
// ======================================================

template <typename Kernel, typename... Args>
float measure_time(Kernel kernel, dim3 grid, dim3 block, size_t shmem, Args... args) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    kernel<<<grid, block, shmem>>>(args...);

    CHECK(cudaGetLastError());        // ← ОБЯЗАТЕЛЬНО
    CHECK(cudaDeviceSynchronize());   // ← ОБЯЗАТЕЛЬНО

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms;
}

// ======================================================
// MAIN
// ======================================================

int main() {
    std::vector<float> h_data(N, 1.0f);
    float *d_a, *d_b, *d_c;

    CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK(cudaMemcpy(d_a, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ================= ЗАДАНИЕ 1 =================
    std::cout << "Задание 1:\n";

    float t1 = measure_time(
    mul_global,
    dim3(blocks), dim3(threads), 0,
    d_a, 2.0f, N
);

    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    float t2 = measure_time(
        mul_shared,
        blocks, threads, threads * sizeof(float),
        d_a, 2.0f, N
    );
    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    std::cout << "Глобальная память: " << t1 << " ms\n";
    std::cout << "Разделяемая память: " << t2 << " ms\n\n";

    // ================= ЗАДАНИЕ 2 =================
    std::cout << "Задание 2:\n";

    int block_sizes[] = {128, 256, 512};
    for (int bs : block_sizes) {
        int bl = (N + bs - 1) / bs;
        float t = measure_time(
            vec_add,
            bl, bs, 0,
            d_a, d_b, d_c, N
        );
        CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel
        std::cout << "Block size " << bs << ": " << t << " ms\n";
    }
    std::cout << "\n";

    // ================= ЗАДАНИЕ 3 =================
    std::cout << "Задание 3:\n";

    float t3 = measure_time(
        coalesced,
        blocks, threads, 0,
        d_a, N
    );
    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    float t4 = measure_time(
        uncoalesced,
        blocks, threads, 0,
        d_a, N
    );
    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    std::cout << "Коалесцированный доступ: " << t3 << " ms\n";
    std::cout << "Некоалесцированный доступ: " << t4 << " ms\n\n";

    // ================= ЗАДАНИЕ 4 =================
    std::cout << "Задание 4:\n";

    float t_bad = measure_time(
        vec_add,
        (N + 64 - 1) / 64, 64, 0,
        d_a, d_b, d_c, N
    );
    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    float t_opt = measure_time(
        vec_add,
        (N + 256 - 1) / 256, 256, 0,
        d_a, d_b, d_c, N
    );
    CHECK(cudaDeviceSynchronize());   // ← ЯВНО после kernel

    std::cout << "Неоптимально (64): " << t_bad << " ms\n";
    std::cout << "Оптимально (256): " << t_opt << " ms\n";

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
