#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Префиксная сумма (Hillis-Steele scan)
__global__ void scan_kernel(float* data, int n) {

    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Загрузка данных в shared memory
    if (idx < n)
        temp[tid] = data[idx];
    else
        temp[tid] = 0.0f;

    __syncthreads();

    // Итеративное суммирование
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float val = 0.0f;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Запись результата
    if (idx < n)
        data[idx] = temp[tid];
}

int main() {

    const int N = 1024;
    const int THREADS = 1024;

    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++)
        h_data[i] = 1.0f;

    float* d_data;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK(cudaMemcpy(d_data, h_data.data(),
                      N * sizeof(float),
                      cudaMemcpyHostToDevice));

    scan_kernel<<<1, THREADS, THREADS * sizeof(float)>>>(d_data, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_data.data(), d_data,
                      N * sizeof(float),
                      cudaMemcpyDeviceToHost));

    std::cout << "Scan result (first 10): ";
    for (int i = 0; i < 10; i++)
        std::cout << h_data[i] << " ";
    std::cout << std::endl;

    cudaFree(d_data);
    return 0;
}
