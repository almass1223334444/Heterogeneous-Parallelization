// ============================================================================
// Task 4: CUDA Merge Sort (GPU)
// Каждая строка закомментирована по требованию задания
// ============================================================================

#include <iostream>                 // cout
#include <vector>                   // std::vector
#include <algorithm>                // std::sort (CPU)
#include <random>                   // генерация случайных чисел
#include <chrono>                   // измерение времени
#include <cuda_runtime.h>           // CUDA runtime API

using namespace std;                // стандартное пространство имён

// ============================================================================
// CUDA kernel: слияние двух отсортированных подмассивов
// ============================================================================

__global__ void mergeKernel(int* input, int* output, int width, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;       // глобальный индекс потока
    int start = idx * (2 * width);                         // начало блока слияния

    if (start >= n) return;                                // если вышли за массив — выходим

    int mid = min(start + width, n);                       // середина подмассива
    int end = min(start + 2 * width, n);                   // конец подмассива

    int i = start;                                         // индекс левого подмассива
    int j = mid;                                           // индекс правого подмассива
    int k = start;                                         // индекс записи результата

    while (i < mid && j < end) {                            // пока оба подмассива не закончились
        if (input[i] <= input[j])                          // сравнение элементов
            output[k++] = input[i++];                      // берём из левого
        else
            output[k++] = input[j++];                      // берём из правого
    }

    while (i < mid) output[k++] = input[i++];              // копируем остаток левого
    while (j < end) output[k++] = input[j++];              // копируем остаток правого
}

// ============================================================================
// GPU merge sort (iterative, bottom-up)
// ============================================================================

void mergeSortGPU(vector<int>& data) {
    int n = data.size();                                   // размер массива
    int* d_input;                                          // указатель на GPU input
    int* d_output;                                         // указатель на GPU output

    cudaMalloc(&d_input, n * sizeof(int));                 // выделяем память на GPU
    cudaMalloc(&d_output, n * sizeof(int));                // выделяем память на GPU

    cudaMemcpy(d_input, data.data(),                        // копируем данные CPU → GPU
               n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;                                     // количество потоков в блоке

    for (int width = 1; width < n; width *= 2) {           // ширина подмассивов
        int blocks = (n + (2 * width * threads) - 1)       // расчёт числа блоков
                     / (2 * width * threads);

        mergeKernel<<<blocks, threads>>>(                  // запуск CUDA kernel
            d_input, d_output, width, n);

        cudaDeviceSynchronize();                            // ждём завершения kernel

        swap(d_input, d_output);                            // меняем input/output местами
    }

    cudaMemcpy(data.data(), d_input,                        // копируем результат GPU → CPU
               n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);                                     // освобождаем память GPU
    cudaFree(d_output);                                    // освобождаем память GPU
}

// ============================================================================
// Генерация массива случайных чисел
// ============================================================================

vector<int> generateArray(int n) {
    vector<int> a(n);                                      // создаём массив
    random_device rd;                                      // источник энтропии
    mt19937 gen(rd());                                     // генератор
    uniform_int_distribution<int> dist(1, 100000);         // диапазон значений

    for (int i = 0; i < n; ++i)                             // цикл заполнения
        a[i] = dist(gen);                                  // случайное число

    return a;                                              // возвращаем массив
}

// ============================================================================
// main
// ============================================================================

int main() {
    cout << "TASK 4 — CUDA Merge Sort\n";                   // заголовок

    vector<int> sizes = {10000, 100000};                   // размеры массивов

    for (int n : sizes) {                                  // цикл по размерам
        cout << "\nArray size: " << n << "\n";             // вывод размера

        vector<int> data = generateArray(n);               // генерация массива
        vector<int> cpuData = data;                        // копия для CPU

        // ---------------- CPU SORT ----------------
        auto t1 = chrono::high_resolution_clock::now();    // старт таймера CPU
        sort(cpuData.begin(), cpuData.end());              // сортировка CPU
        auto t2 = chrono::high_resolution_clock::now();    // конец таймера CPU

        double cpuTime = chrono::duration<double, milli>(t2 - t1).count();

        // ---------------- GPU SORT ----------------
        auto t3 = chrono::high_resolution_clock::now();    // старт таймера GPU
        mergeSortGPU(data);                                // сортировка GPU
        auto t4 = chrono::high_resolution_clock::now();    // конец таймера GPU

        double gpuTime = chrono::duration<double, milli>(t4 - t3).count();

        // ---------------- OUTPUT ----------------
        cout << "CPU time: " << cpuTime << " ms\n";        // вывод CPU времени
        cout << "GPU time: " << gpuTime << " ms\n";        // вывод GPU времени
        cout << "Speedup: " << cpuTime / gpuTime << "x\n"; // ускорение
    }



    return 0;                                              // завершение программы
}
