#include <iostream> // подключаем библиотеку для ввода-вывода
#include <vector> // подключаем библиотеку для работы с динамическими массивами
#include <omp.h> // подключаем библиотеку openmp для многопоточности на cpu
#include <cuda_runtime.h> // подключаем библиотеку cuda для работы с gpu
#include <chrono> // подключаем библиотеку для замера времени выполнения
#include <thread> // подключаем библиотеку для работы с потоками

// это ядро cuda - функция, которая выполняется на gpu
__global__ void multiplyByTwo(float* data, int N) {
    // вычисляем глобальный индекс текущего потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем, что индекс не выходит за границы массива
    if (i < N) {
        data[i] = data[i] * 2; // умножаем элемент на 2
    }
}

// функция для обработки части массива на cpu
void processCPU(std::vector<float>& data, int start, int end) {
    // используем openmp для параллельной обработки части массива
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        data[i] = data[i] * 2; // умножаем каждый элемент на 2
    }
}

// функция для обработки части массива на gpu
void processGPU(std::vector<float>& data, int start, int end) {
    // вычисляем размер обрабатываемой части массива
    int N = end - start;
    
    // вычисляем размер части массива в байтах
    size_t size = N * sizeof(float);
    
    // объявляем указатель для массива на gpu
    float* d_data;
    
    // выделяем память на gpu
    cudaMalloc(&d_data, size);
    
    // копируем часть массива с cpu на gpu
    // data.data() + start - указатель на начало обрабатываемой части
    cudaMemcpy(d_data, data.data() + start, size, cudaMemcpyHostToDevice);
    
    // определяем конфигурацию запуска ядра
    int threadsPerBlock = 256; // количество потоков в одном блоке
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // количество блоков
    
    // запускаем ядро на gpu
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    
    // ждем завершения всех операций на gpu
    cudaDeviceSynchronize();
    
    // копируем обработанные данные обратно на cpu
    cudaMemcpy(data.data() + start, d_data, size, cudaMemcpyDeviceToHost);
    
    // освобождаем память на gpu
    cudaFree(d_data);
}

int main() {
    // задаем размер массива - миллион элементов
    const int N = 1000000;
    
    // создаем массив на cpu и заполняем его значениями
    std::vector<float> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i; // каждый элемент равен своему индексу
    }
    
    // вычисляем точку разделения массива (середина)
    int mid = N / 2;
    
    // запоминаем время начала обработки
    auto start = std::chrono::high_resolution_clock::now();
    
    // создаем два потока для одновременной обработки на cpu и gpu
    // первый поток обрабатывает первую половину массива на cpu
    std::thread cpuThread(processCPU, std::ref(data), 0, mid);
    
    // второй поток обрабатывает вторую половину массива на gpu
    std::thread gpuThread(processGPU, std::ref(data), mid, N);
    
    // ждем завершения обработки на cpu
    cpuThread.join();
    
    // ждем завершения обработки на gpu
    gpuThread.join();
    
    // запоминаем время окончания обработки
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем разницу во времени и переводим в миллисекунды
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // выводим результаты на экран
    std::cout << "гибридная обработка (cpu + gpu):" << std::endl;
    std::cout << "размер массива: " << N << std::endl;
    std::cout << "cpu обработало: " << mid << " элементов" << std::endl;
    std::cout << "gpu обработало: " << (N - mid) << " элементов" << std::endl;
    std::cout << "время выполнения: " << duration.count() << " мс" << std::endl;
    
    // выводим несколько первых элементов из каждой части для проверки
    std::cout << "первые 3 элемента (cpu часть): ";
    for (int i = 0; i < 3; i++) {
        std::cout << data[i] << " "; // элементы должны быть 0, 2, 4
    }
    std::cout << std::endl;
    
    std::cout << "первые 3 элемента (gpu часть): ";
    for (int i = mid; i < mid + 3; i++) {
        std::cout << data[i] << " "; // элементы должны быть удвоенными
    }
    std::cout << std::endl;
    
    return 0; // завершаем программу
}
