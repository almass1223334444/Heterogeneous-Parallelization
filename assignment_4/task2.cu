#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 1024                 // Упрощённый размер массива (для одного блока)

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// CPU scan
void cpu_scan(const std::vector<float>& in, std::vector<float>& out) {
    float sum = 0.0f;                 // Инициализация суммы
    for (int i = 0; i < in.size(); i++) {
        sum += in[i];                 // Накопление суммы
        out[i] = sum;                 // Запись результата
    }
}

// GPU scan (Blelloch)
__global__ void scan_shared(float* in, float* out) {
    __shared__ float temp[1024];      // Shared память
    int tid = threadIdx.x;            // Локальный индекс внутри блока

    temp[tid] = in[tid];              // Загружаем данные в shared
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;             // Накопление
        __syncthreads();
    }

    out[tid] = temp[tid];             // Записываем результат
}

int main() {
    std::vector<float> h_in(N, 1.0f);
    std::vector<float> h_out(N);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_scan(h_in, h_out);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    scan_shared<<<1, N>>>(d_in, d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpuTime;
    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";

    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    return 0;
}
