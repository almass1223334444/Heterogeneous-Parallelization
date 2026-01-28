#include <iostream> // подключаем библиотеку для ввода-вывода
#include <vector> // подключаем библиотеку для работы с динамическими массивами
#include <cuda_runtime.h> // подключаем библиотеку cuda для работы с gpu
#include <chrono> // подключаем библиотеку для замера времени выполнения

// это ядро cuda - функция, которая выполняется на gpu
// __global__ означает, что функция вызывается с cpu, но выполняется на gpu
__global__ void multiplyByTwo(float* data, int N) {
    // вычисляем глобальный индекс текущего потока
    // blockIdx.x - номер блока, blockDim.x - размер блока, threadIdx.x - номер потока в блоке
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем, что индекс не выходит за границы массива
    if (i < N) {
        data[i] = data[i] * 2; // умножаем элемент на 2
    }
}

int main() {
    // задаем размер массива - миллион элементов
    const int N = 1000000;
    
    // вычисляем размер массива в байтах (каждый float занимает 4 байта)
    size_t size = N * sizeof(float);
    
    // создаем массив на cpu (host) и заполняем его значениями
    std::vector<float> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = i; // каждый элемент равен своему индексу
    }
    
    // объявляем указатель для массива на gpu (device)
    float* d_data;
    
    // выделяем память на gpu размером size байт
    cudaMalloc(&d_data, size);
    
    // запоминаем время начала обработки (включая копирование данных)
    auto start = std::chrono::high_resolution_clock::now();
    
    // копируем данные с cpu (host) на gpu (device)
    // h_data.data() - указатель на начало массива на cpu
    // d_data - указатель на массив на gpu
    // cudaMemcpyHostToDevice - направление копирования: с cpu на gpu
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
    
    // определяем конфигурацию запуска ядра
    int threadsPerBlock = 256; // количество потоков в одном блоке
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // количество блоков (округляем вверх)
    
    // запускаем ядро на gpu
    // <<<blocksPerGrid, threadsPerBlock>>> - синтаксис cuda для указания конфигурации
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    
    // ждем завершения всех операций на gpu
    cudaDeviceSynchronize();
    
    // копируем обработанные данные обратно с gpu на cpu
    // cudaMemcpyDeviceToHost - направление копирования: с gpu на cpu
    cudaMemcpy(h_data.data(), d_data, size, cudaMemcpyDeviceToHost);
    
    // запоминаем время окончания обработки
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем разницу во времени и переводим в миллисекунды
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // освобождаем память на gpu
    cudaFree(d_data);
    
    // выводим результаты на экран
    std::cout << "обработка на gpu (cuda):" << std::endl;
    std::cout << "размер массива: " << N << std::endl;
    std::cout << "время выполнения: " << duration.count() << " мс" << std::endl;
    
    // выводим несколько первых элементов для проверки корректности
    std::cout << "первые 5 элементов после обработки: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_data[i] << " "; // элементы должны быть 0, 2, 4, 6, 8
    }
    std::cout << std::endl;
    
    return 0; // завершаем программу
}
