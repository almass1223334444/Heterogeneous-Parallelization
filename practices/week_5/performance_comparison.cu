#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <stack>
#include <queue>

// структура параллельного стека (из основной программы)
struct ParallelStack {
    int *data;
    int top;
    int capacity;

    __device__ void init(int *buffer, int size) {
        data = buffer;
        top = -1;
        capacity = size;
    }

    __device__ bool push(int value) {
        int pos = atomicAdd(&top, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool pop(int *value) {
        int pos = atomicSub(&top, 1);
        if (pos >= 0) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

// структура параллельной очереди (из основной программы)
struct ParallelQueue {
    int *data;
    int head;
    int tail;
    int capacity;

    __device__ void init(int *buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }
        return false;
    }

    __device__ bool dequeue(int *value) {
        int pos = atomicAdd(&head, 1);
        if (pos < tail) {
            *value = data[pos];
            return true;
        }
        return false;
    }
};

// ядро для параллельного стека
__global__ void parallelStackKernel(int *buffer, int size, int numOperations) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // создаем объект стека
    __shared__ ParallelStack stack;
    
    // инициализируем стек (только первый поток)
    if (threadIdx.x == 0) {
        stack.init(buffer, size);
    }
    __syncthreads();
    
    // каждый поток выполняет операции push
    for (int i = 0; i < numOperations; i++) {
        stack.push(tid * 1000 + i);
    }
    
    __syncthreads();
    
    // каждый поток выполняет операции pop
    for (int i = 0; i < numOperations; i++) {
        int value;
        stack.pop(&value);
    }
}

// ядро для параллельной очереди
__global__ void parallelQueueKernel(int *buffer, int size, int numOperations) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // создаем объект очереди
    __shared__ ParallelQueue queue;
    
    // инициализируем очередь (только первый поток)
    if (threadIdx.x == 0) {
        queue.init(buffer, size);
    }
    __syncthreads();
    
    // каждый поток выполняет операции enqueue
    for (int i = 0; i < numOperations; i++) {
        queue.enqueue(tid * 1000 + i);
    }
    
    __syncthreads();
    
    // каждый поток выполняет операции dequeue
    for (int i = 0; i < numOperations; i++) {
        int value;
        queue.dequeue(&value);
    }
}

// функция для проверки ошибок cuda
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA ошибка: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// функция для тестирования последовательного стека
double testSequentialStack(int numOperations) {
    // создаем стандартный стек из stl
    std::stack<int> stack;
    
    // замеряем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // выполняем операции push
    for (int i = 0; i < numOperations; i++) {
        stack.push(i);
    }
    
    // выполняем операции pop
    for (int i = 0; i < numOperations; i++) {
        if (!stack.empty()) {
            stack.pop();
        }
    }
    
    // замеряем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем длительность в микросекундах
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

// функция для тестирования последовательной очереди
double testSequentialQueue(int numOperations) {
    // создаем стандартную очередь из stl
    std::queue<int> queue;
    
    // замеряем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // выполняем операции push
    for (int i = 0; i < numOperations; i++) {
        queue.push(i);
    }
    
    // выполняем операции pop
    for (int i = 0; i < numOperations; i++) {
        if (!queue.empty()) {
            queue.pop();
        }
    }
    
    // замеряем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // вычисляем длительность в микросекундах
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

// функция для тестирования параллельного стека
double testParallelStack(int bufferSize, int numThreads, int numOperations) {
    // выделяем память на устройстве для буфера
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, bufferSize * sizeof(int)),
        "выделение памяти для буфера стека"
    );
    
    // замеряем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // запускаем ядро
    parallelStackKernel<<<1, numThreads>>>(d_buffer, bufferSize, numOperations);
    
    // синхронизируем для точного замера времени
    cudaDeviceSynchronize();
    
    // замеряем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // проверяем ошибки
    checkCudaError(cudaGetLastError(), "запуск ядра параллельного стека");
    
    // освобождаем память
    cudaFree(d_buffer);
    
    // вычисляем длительность в микросекундах
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

// функция для тестирования параллельной очереди
double testParallelQueue(int bufferSize, int numThreads, int numOperations) {
    // выделяем память на устройстве для буфера
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, bufferSize * sizeof(int)),
        "выделение памяти для буфера очереди"
    );
    
    // замеряем время начала
    auto start = std::chrono::high_resolution_clock::now();
    
    // запускаем ядро
    parallelQueueKernel<<<1, numThreads>>>(d_buffer, bufferSize, numOperations);
    
    // синхронизируем для точного замера времени
    cudaDeviceSynchronize();
    
    // замеряем время окончания
    auto end = std::chrono::high_resolution_clock::now();
    
    // проверяем ошибки
    checkCudaError(cudaGetLastError(), "запуск ядра параллельной очереди");
    
    // освобождаем память
    cudaFree(d_buffer);
    
    // вычисляем длительность в микросекундах
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count();
}

// функция для сравнения производительности
void comparePerformance() {
    std::cout << "\n=== Сравнение производительности ===" << std::endl;
    
    // параметры тестирования
    const int bufferSize = 10000;      // размер буфера
    const int numThreads = 128;        // количество потоков для параллельной версии
    const int numOperations = 100;     // операций на поток
    const int totalOps = numThreads * numOperations;  // общее количество операций
    
    std::cout << "\nПараметры теста:" << std::endl;
    std::cout << "- Размер буфера: " << bufferSize << std::endl;
    std::cout << "- Количество потоков: " << numThreads << std::endl;
    std::cout << "- Операций на поток: " << numOperations << std::endl;
    std::cout << "- Всего операций: " << totalOps << std::endl;
    
    // тестируем последовательный стек
    std::cout << "\n--- Стек ---" << std::endl;
    double seqStackTime = testSequentialStack(totalOps);
    std::cout << "Последовательный стек: " << seqStackTime << " микросекунд" << std::endl;
    
    // тестируем параллельный стек
    double parStackTime = testParallelStack(bufferSize, numThreads, numOperations);
    std::cout << "Параллельный стек: " << parStackTime << " микросекунд" << std::endl;
    
    // вычисляем ускорение для стека
    double stackSpeedup = seqStackTime / parStackTime;
    std::cout << "Ускорение стека: " << stackSpeedup << "x" << std::endl;
    
    // тестируем последовательную очередь
    std::cout << "\n--- Очередь ---" << std::endl;
    double seqQueueTime = testSequentialQueue(totalOps);
    std::cout << "Последовательная очередь: " << seqQueueTime << " микросекунд" << std::endl;
    
    // тестируем параллельную очередь
    double parQueueTime = testParallelQueue(bufferSize, numThreads, numOperations);
    std::cout << "Параллельная очередь: " << parQueueTime << " микросекунд" << std::endl;
    
    // вычисляем ускорение для очереди
    double queueSpeedup = seqQueueTime / parQueueTime;
    std::cout << "Ускорение очереди: " << queueSpeedup << "x" << std::endl;
    
    // сравниваем стек и очередь между собой
    std::cout << "\n--- Сравнение стека и очереди ---" << std::endl;
    if (parStackTime < parQueueTime) {
        double diff = (parQueueTime / parStackTime - 1) * 100;
        std::cout << "Параллельный стек быстрее очереди на " << diff << "%" << std::endl;
    } else {
        double diff = (parStackTime / parQueueTime - 1) * 100;
        std::cout << "Параллельная очередь быстрее стека на " << diff << "%" << std::endl;
    }
    
    // выводим выводы
    std::cout << "\n--- Выводы ---" << std::endl;
    std::cout << "1. Параллельные структуры данных ";
    if (stackSpeedup > 1 && queueSpeedup > 1) {
        std::cout << "показывают ускорение по сравнению с последовательными" << std::endl;
    } else {
        std::cout << "могут быть медленнее из-за overhead синхронизации" << std::endl;
    }
    
    std::cout << "2. Атомарные операции добавляют накладные расходы," << std::endl;
    std::cout << "   но позволяют безопасную параллельную работу" << std::endl;
    
    std::cout << "3. Эффективность зависит от количества потоков," << std::endl;
    std::cout << "   размера данных и типа операций" << std::endl;
}

// главная функция программы
int main() {
    std::cout << "Программа сравнения производительности структур данных" << std::endl;
    
    // проводим сравнение
    comparePerformance();
    
    std::cout << "\nПрограмма завершена успешно!" << std::endl;
    return 0;
}
