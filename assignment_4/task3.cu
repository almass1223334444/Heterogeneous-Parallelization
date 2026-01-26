#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 1000000               // Размер массива
#define HALF (N/2)              // Половина массива

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// CPU сумма
float cpu_sum(const float* data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

// GPU ядро (atomicAdd)
__global__ void sum_gpu(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(result, data[idx]);
}

int main() {
    std::vector<float> h_data(N, 1.0f);          // Заполняем массив единицами

    // CPU-only
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpuRes = cpu_sum(h_data.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU-only
    float *d_data, *d_res;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK(cudaMalloc(&d_res, sizeof(float)));
    CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res, 0, sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    sum_gpu<<<blocks, threads>>>(d_data, d_res, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float gpuTime;
    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    float gpuRes;
    CHECK(cudaMemcpy(&gpuRes, d_res, sizeof(float), cudaMemcpyDeviceToHost));

    // Hybrid (CPU + GPU параллельно)
    float *d_data2, *d_res2;
    CHECK(cudaMalloc(&d_data2, HALF * sizeof(float)));
    CHECK(cudaMalloc(&d_res2, sizeof(float)));
    CHECK(cudaMemcpy(d_data2, h_data.data() + HALF, HALF * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_res2, 0, sizeof(float)));

    auto th0 = std::chrono::high_resolution_clock::now();

    // CPU часть
    float cpuPart = cpu_sum(h_data.data(), HALF);

    // GPU часть
    sum_gpu<<<blocks/2, threads>>>(d_data2, d_res2, HALF);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    auto th1 = std::chrono::high_resolution_clock::now();
    double hybridTime = std::chrono::duration<double, std::milli>(th1 - th0).count();

    float gpuPart;
    CHECK(cudaMemcpy(&gpuPart, d_res2, sizeof(float), cudaMemcpyDeviceToHost));

    float hybridRes = cpuPart + gpuPart;

    std::cout << "CPU time: " << cpuTime << " ms, result=" << cpuRes << "\n";
    std::cout << "GPU time: " << gpuTime << " ms, result=" << gpuRes << "\n";
    std::cout << "Hybrid time: " << hybridTime << " ms, result=" << hybridRes << "\n";

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_data2));
    CHECK(cudaFree(d_res2));

    return 0;
}
