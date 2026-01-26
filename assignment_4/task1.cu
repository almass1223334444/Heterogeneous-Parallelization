#include <cuda_runtime.h>        // Подключаем CUDA runtime API
#include <iostream>              // Для ввода/вывода
#include <vector>                // Для std::vector
#include <chrono>                // Для измерения времени

#define N 100000                 // Размер массива 100000 элементов

// Макрос для проверки ошибок CUDA
#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        exit(1); \
    } \
}

// CPU-суммирование
float cpu_sum(const std::vector<float>& data) {
    float sum = 0.0f;                   // Инициализируем сумму
    for (float x : data)                // Проходим по всем элементам массива
        sum += x;                       // Добавляем элемент к сумме
    return sum;                         // Возвращаем результат
}

// CUDA-ядро: суммирование через atomicAdd в глобальной памяти
__global__ void sum_global(float* data, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Вычисляем глобальный индекс
    if (idx < n) {                                    // Проверяем выход за пределы
        atomicAdd(result, data[idx]);                 // Добавляем элемент в общий результат
    }
}

int main() {
    // Создаем массив на CPU
    std::vector<float> h_data(N, 1.0f);               // Заполняем массив единицами

    // Выделяем память на GPU
    float *d_data, *d_result;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));     // Выделяем память для данных
    CHECK(cudaMalloc(&d_result, sizeof(float)));       // Выделяем память для результата

    // Копируем данные на GPU
    CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU вычисление
    auto start_cpu = std::chrono::high_resolution_clock::now(); // Запуск таймера CPU
    float cpuRes = cpu_sum(h_data);                              // Суммируем на CPU
    auto end_cpu = std::chrono::high_resolution_clock::now();   // Остановка таймера
    double cpuTime = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // GPU вычисление
    CHECK(cudaMemset(d_result, 0, sizeof(float)));               // Обнуляем результат на GPU

    int threads = 256;                                           // Размер блока
    int blocks = (N + threads - 1) / threads;                    // Количество блоков

    cudaEvent_t start, stop;                                     // Таймер CUDA
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));                               // Запускаем таймер GPU
    sum_global<<<blocks, threads>>>(d_data, d_result, N);        // Запускаем ядро
    CHECK(cudaGetLastError());                                   // Проверка ошибок ядра
    CHECK(cudaEventRecord(stop));                                // Останавливаем таймер
    CHECK(cudaEventSynchronize(stop));                           // Синхронизация

    float gpuTime;
    CHECK(cudaEventElapsedTime(&gpuTime, start, stop));          // Время GPU

    float gpuRes;
    CHECK(cudaMemcpy(&gpuRes, d_result, sizeof(float), cudaMemcpyDeviceToHost)); // Копируем результат на CPU

    // Вывод результатов
    std::cout << "CPU sum: " << cpuRes << "\n";
    std::cout << "GPU sum: " << gpuRes << "\n";
    std::cout << "CPU time: " << cpuTime << " ms\n";
    std::cout << "GPU time: " << gpuTime << " ms\n";

    // Освобождаем память
    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_result));

    return 0;
}
