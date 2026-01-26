#include <cuda_runtime.h>      // Основные CUDA-функции
#include <iostream>            // Ввод / вывод
#include <vector>              // Контейнер vector

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " \
                  << cudaGetErrorString(err) \
                  << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// CUDA-ядро редукции
__global__ void reduction_kernel(float* input, float* output, int n) {

    // Разделяемая память для блока
    extern __shared__ float sdata[];

    // Индекс потока внутри блока   
    int tid = threadIdx.x;

    // Глобальный индекс элемента
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Загружаем данные из глобальной памяти в shared
    if (idx < n)
        sdata[tid] = input[idx];
    else
        sdata[tid] = 0.0f;

    __syncthreads(); // Синхронизация потоков блока

    // Параллельная редукция внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Первый поток блока записывает результат
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {

    const int N = 1024;                   // Размер массива
    const int THREADS = 256;              // Потоки в блоке
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    // Хост-массив
    std::vector<float> h_input(N, 1.0f);

    // Указатели на устройство
    float *d_input, *d_output;

    // Выделяем память на GPU
    CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK(cudaMalloc(&d_output, BLOCKS * sizeof(float)));

    // Копируем данные на GPU
    CHECK(cudaMemcpy(d_input, h_input.data(),
                      N * sizeof(float),
                      cudaMemcpyHostToDevice));

    // Запуск ядра
    reduction_kernel<<<BLOCKS, THREADS, THREADS * sizeof(float)>>>(
        d_input, d_output, N);

    CHECK(cudaDeviceSynchronize());

    // Копируем частичные суммы обратно
    std::vector<float> h_output(BLOCKS);
    CHECK(cudaMemcpy(h_output.data(), d_output,
                      BLOCKS * sizeof(float),
                      cudaMemcpyDeviceToHost));

    // Финальная сумма на CPU
    float sum = 0.0f;
    for (float x : h_output)
        sum += x;

    std::cout << "Reduction result = " << sum << std::endl;

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
