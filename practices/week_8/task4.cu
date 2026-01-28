#include <iostream> // подключаем библиотеку для ввода-вывода
#include <vector> // подключаем библиотеку для работы с динамическими массивами
#include <omp.h> // подключаем библиотеку openmp для многопоточности на cpu
#include <cuda_runtime.h> // подключаем библиотеку cuda для работы с gpu
#include <chrono> // подключаем библиотеку для замера времени выполнения
#include <thread> // подключаем библиотеку для работы с потоками
#include <iomanip> // подключаем библиотеку для форматирования вывода

// ядро cuda для обработки на gpu
__global__ void multiplyByTwo(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = data[i] * 2;
    }
}

// функция для обработки массива на cpu с openmp
double processCPUOnly(std::vector<float>& data) {
    int N = data.size(); // получаем размер массива
    
    // запоминаем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // параллельная обработка на cpu
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        data[i] = data[i] * 2;
    }
    
    // запоминаем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем время в миллисекундах и возвращаем
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

// функция для обработки массива на gpu с cuda
double processGPUOnly(std::vector<float>& data) {
    int N = data.size(); // получаем размер массива
    size_t size = N * sizeof(float); // вычисляем размер в байтах
    
    float* d_data; // указатель для массива на gpu
    cudaMalloc(&d_data, size); // выделяем память на gpu
    
    // запоминаем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // копируем данные на gpu
    cudaMemcpy(d_data, data.data(), size, cudaMemcpyHostToDevice);
    
    // определяем конфигурацию запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // запускаем ядро
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    
    // ждем завершения
    cudaDeviceSynchronize();
    
    // копируем данные обратно
    cudaMemcpy(data.data(), d_data, size, cudaMemcpyDeviceToHost);
    
    // запоминаем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // освобождаем память
    cudaFree(d_data);
    
    // вычисляем время в миллисекундах и возвращаем
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

// функция для обработки части массива на cpu (для гибридного режима)
void processCPUPart(std::vector<float>& data, int start, int end) {
    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        data[i] = data[i] * 2;
    }
}

// функция для обработки части массива на gpu (для гибридного режима)
void processGPUPart(std::vector<float>& data, int start, int end) {
    int N = end - start;
    size_t size = N * sizeof(float);
    
    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, data.data() + start, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(data.data() + start, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// функция для гибридной обработки массива
double processHybrid(std::vector<float>& data) {
    int N = data.size(); // получаем размер массива
    int mid = N / 2; // находим середину массива
    
    // запоминаем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // создаем два потока для одновременной работы
    std::thread cpuThread(processCPUPart, std::ref(data), 0, mid);
    std::thread gpuThread(processGPUPart, std::ref(data), mid, N);
    
    // ждем завершения обоих потоков
    cpuThread.join();
    gpuThread.join();
    
    // запоминаем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем время в миллисекундах и возвращаем
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main() {
    // задаем размер массива
    const int N = 1000000;
    
    std::cout << "========================================" << std::endl;
    std::cout << "анализ производительности" << std::endl;
    std::cout << "размер массива: " << N << " элементов" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    // переменные для хранения времени выполнения
    double timeCPU, timeGPU, timeHybrid;
    
    // тест 1: обработка на cpu
    {
        std::vector<float> data(N);
        for (int i = 0; i < N; i++) data[i] = i;
        
        timeCPU = processCPUOnly(data);
        
        std::cout << "1. обработка на cpu (openmp):" << std::endl;
        std::cout << "   время: " << std::fixed << std::setprecision(3) << timeCPU << " мс" << std::endl;
        std::cout << "   проверка: первые 3 элемента = " << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
        std::cout << std::endl;
    }
    
    // тест 2: обработка на gpu
    {
        std::vector<float> data(N);
        for (int i = 0; i < N; i++) data[i] = i;
        
        timeGPU = processGPUOnly(data);
        
        std::cout << "2. обработка на gpu (cuda):" << std::endl;
        std::cout << "   время: " << std::fixed << std::setprecision(3) << timeGPU << " мс" << std::endl;
        std::cout << "   проверка: первые 3 элемента = " << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
        std::cout << std::endl;
    }
    
    // тест 3: гибридная обработка
    {
        std::vector<float> data(N);
        for (int i = 0; i < N; i++) data[i] = i;
        
        timeHybrid = processHybrid(data);
        
        std::cout << "3. гибридная обработка (cpu + gpu):" << std::endl;
        std::cout << "   время: " << std::fixed << std::setprecision(3) << timeHybrid << " мс" << std::endl;
        std::cout << "   проверка cpu части: " << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
        std::cout << "   проверка gpu части: " << data[N/2] << ", " << data[N/2+1] << ", " << data[N/2+2] << std::endl;
        std::cout << std::endl;
    }
    
    // выводим сравнительный анализ
    std::cout << "========================================" << std::endl;
    std::cout << "сравнительный анализ:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // находим самый быстрый метод
    double minTime = std::min({timeCPU, timeGPU, timeHybrid});
    
    // вычисляем ускорение относительно самого быстрого
    std::cout << "cpu:     " << std::fixed << std::setprecision(3) << timeCPU << " мс (x" 
              << std::setprecision(2) << timeCPU/minTime << ")" << std::endl;
    std::cout << "gpu:     " << std::fixed << std::setprecision(3) << timeGPU << " мс (x" 
              << std::setprecision(2) << timeGPU/minTime << ")" << std::endl;
    std::cout << "гибрид:  " << std::fixed << std::setprecision(3) << timeHybrid << " мс (x" 
              << std::setprecision(2) << timeHybrid/minTime << ")" << std::endl;
    std::cout << std::endl;
    
    // выводим выводы
    std::cout << "выводы:" << std::endl;
    
    if (timeHybrid < timeCPU && timeHybrid < timeGPU) {
        std::cout << "гибридный подход показал лучшую производительность" << std::endl;
        std::cout << "ускорение относительно cpu: x" << std::setprecision(2) << timeCPU/timeHybrid << std::endl;
        std::cout << "ускорение относительно gpu: x" << std::setprecision(2) << timeGPU/timeHybrid << std::endl;
    } else if (timeGPU < timeCPU && timeGPU < timeHybrid) {
        std::cout << "gpu показал лучшую производительность" << std::endl;
        std::cout << "гибридный подход проигрывает из-за накладных расходов на разделение работы" << std::endl;
    } else {
        std::cout << "cpu показал лучшую производительность" << std::endl;
        std::cout << "массив может быть слишком мал для эффективного использования gpu" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "факторы, влияющие на производительность:" << std::endl;
    std::cout << "- размер массива (чем больше, тем выгоднее gpu)" << std::endl;
    std::cout << "- время передачи данных между cpu и gpu" << std::endl;
    std::cout << "- сложность операций (простые операции могут быть невыгодны для gpu)" << std::endl;
    std::cout << "- баланс нагрузки между cpu и gpu в гибридном режиме" << std::endl;
    
    return 0; // завершаем программу
}
