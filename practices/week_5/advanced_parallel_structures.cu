#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// структура очереди с поддержкой нескольких производителей и потребителей (mpmc)
struct MPMCQueue {
    int *data;          // указатель на массив данных очереди
    int *head;          // указатель на индекс начала очереди
    int *tail;          // указатель на индекс конца очереди
    int capacity;       // максимальная вместимость очереди
    int *size;          // текущее количество элементов в очереди

    // инициализация очереди с буфером и размером
    __device__ void init(int *buffer, int *headPtr, int *tailPtr, int *sizePtr, int cap) {
        data = buffer;      // присваиваем указатель на буфер данных
        head = headPtr;     // присваиваем указатель на head
        tail = tailPtr;     // присваиваем указатель на tail
        size = sizePtr;     // присваиваем указатель на size
        capacity = cap;     // сохраняем максимальный размер
        
        // инициализируем начальные значения
        *head = 0;          // начало очереди
        *tail = 0;          // конец очереди
        *size = 0;          // количество элементов
    }

    // добавление элемента в очередь (thread-safe для множественных производителей)
    __device__ bool enqueue(int value) {
        // атомарно проверяем, есть ли место в очереди
        int currentSize = atomicAdd(size, 1);
        
        // если очередь переполнена, откатываем операцию
        if (currentSize >= capacity) {
            atomicSub(size, 1);  // возвращаем size обратно
            return false;         // очередь полна
        }
        
        // атомарно получаем позицию для записи
        int pos = atomicAdd(tail, 1) % capacity;
        
        // записываем значение в очередь
        data[pos] = value;
        
        return true;  // успешное добавление
    }

    // извлечение элемента из очереди (thread-safe для множественных потребителей)
    __device__ bool dequeue(int *value) {
        // атомарно проверяем, есть ли элементы в очереди
        int currentSize = atomicSub(size, 1);
        
        // если очередь пуста, откатываем операцию
        if (currentSize <= 0) {
            atomicAdd(size, 1);  // возвращаем size обратно
            return false;         // очередь пуста
        }
        
        // атомарно получаем позицию для чтения
        int pos = atomicAdd(head, 1) % capacity;
        
        // читаем значение из очереди
        *value = data[pos];
        
        return true;  // успешное извлечение
    }
};

// ядро для тестирования mpmc очереди с производителями и потребителями
__global__ void testMPMCKernel(int *buffer, int *head, int *tail, int *size, 
                                int capacity, int *producerResults, int *consumerResults,
                                int numOperations, bool isProducer) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // создаем локальный объект очереди
    __shared__ MPMCQueue queue;
    
    // только первый поток в блоке инициализирует очередь
    if (threadIdx.x == 0) {
        queue.init(buffer, head, tail, size, capacity);
    }
    // синхронизируем потоки
    __syncthreads();
    
    // если это поток-производитель
    if (isProducer) {
        // пытаемся добавить элементы в очередь
        for (int i = 0; i < numOperations; i++) {
            int value = tid * 1000 + i;  // уникальное значение
            bool success = queue.enqueue(value);
            producerResults[tid * numOperations + i] = success ? 1 : 0;  // сохраняем результат
        }
    } 
    // если это поток-потребитель
    else {
        // пытаемся извлечь элементы из очереди
        for (int i = 0; i < numOperations; i++) {
            int value;
            bool success = queue.dequeue(&value);
            if (success) {
                consumerResults[tid * numOperations + i] = value;  // сохраняем значение
            } else {
                consumerResults[tid * numOperations + i] = -1;  // помечаем неудачу
            }
        }
    }
}

// ядро с использованием разделяемой памяти для оптимизации
__global__ void testSharedMemoryQueue(int *globalBuffer, int capacity, 
                                      int *results, int numOperations) {
    // выделяем разделяемую память для буфера очереди
    extern __shared__ int sharedBuffer[];
    
    // выделяем память для метаданных очереди
    __shared__ int sharedHead;
    __shared__ int sharedTail;
    __shared__ int sharedSize;
    
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // инициализируем очередь (только первый поток)
    if (threadIdx.x == 0) {
        sharedHead = 0;
        sharedTail = 0;
        sharedSize = 0;
    }
    // синхронизируем потоки
    __syncthreads();
    
    // каждый поток добавляет данные в разделяемую память
    for (int i = 0; i < numOperations; i++) {
        // атомарно проверяем размер
        int currentSize = atomicAdd(&sharedSize, 1);
        
        if (currentSize < capacity) {
            // атомарно получаем позицию
            int pos = atomicAdd(&sharedTail, 1) % capacity;
            // записываем в разделяемую память
            sharedBuffer[pos] = tid * 1000 + i;
        } else {
            atomicSub(&sharedSize, 1);  // откатываем
        }
    }
    
    // синхронизируем перед чтением
    __syncthreads();
    
    // каждый поток читает данные из разделяемой памяти
    for (int i = 0; i < numOperations; i++) {
        int currentSize = atomicSub(&sharedSize, 1);
        
        if (currentSize > 0) {
            // атомарно получаем позицию для чтения
            int pos = atomicAdd(&sharedHead, 1) % capacity;
            // читаем из разделяемой памяти
            results[tid * numOperations + i] = sharedBuffer[pos];
        } else {
            atomicAdd(&sharedSize, 1);  // откатываем
            results[tid * numOperations + i] = -1;
        }
    }
    
    // синхронизируем перед копированием в глобальную память
    __syncthreads();
    
    // копируем результаты в глобальную память (для демонстрации)
    if (threadIdx.x < capacity && threadIdx.x < sharedSize) {
        globalBuffer[threadIdx.x] = sharedBuffer[threadIdx.x];
    }
}

// функция для проверки ошибок cuda
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA ошибка: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// функция для тестирования mpmc очереди
void testMPMCQueue() {
    std::cout << "\n=== Тестирование MPMC очереди ===" << std::endl;
    
    const int capacity = 512;           // размер буфера очереди
    const int numProducers = 16;        // количество производителей
    const int numConsumers = 16;        // количество потребителей
    const int numOperations = 10;       // операций на поток
    
    // выделяем память на устройстве для буфера
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, capacity * sizeof(int)),
        "выделение памяти для буфера mpmc"
    );
    
    // выделяем память для метаданных очереди
    int *d_head, *d_tail, *d_size;
    checkCudaError(cudaMalloc(&d_head, sizeof(int)), "выделение памяти для head");
    checkCudaError(cudaMalloc(&d_tail, sizeof(int)), "выделение памяти для tail");
    checkCudaError(cudaMalloc(&d_size, sizeof(int)), "выделение памяти для size");
    
    // выделяем память для результатов производителей
    int *d_producerResults;
    checkCudaError(
        cudaMalloc(&d_producerResults, numProducers * numOperations * sizeof(int)),
        "выделение памяти для результатов производителей"
    );
    
    // выделяем память для результатов потребителей
    int *d_consumerResults;
    checkCudaError(
        cudaMalloc(&d_consumerResults, numConsumers * numOperations * sizeof(int)),
        "выделение памяти для результатов потребителей"
    );
    
    // запускаем производителей
    auto start = std::chrono::high_resolution_clock::now();
    testMPMCKernel<<<1, numProducers>>>(d_buffer, d_head, d_tail, d_size, capacity,
                                        d_producerResults, d_consumerResults, 
                                        numOperations, true);
    cudaDeviceSynchronize();
    
    // запускаем потребителей
    testMPMCKernel<<<1, numConsumers>>>(d_buffer, d_head, d_tail, d_size, capacity,
                                        d_producerResults, d_consumerResults, 
                                        numOperations, false);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    checkCudaError(cudaGetLastError(), "запуск ядра mpmc");
    
    // копируем результаты на хост
    int *h_producerResults = new int[numProducers * numOperations];
    int *h_consumerResults = new int[numConsumers * numOperations];
    
    checkCudaError(
        cudaMemcpy(h_producerResults, d_producerResults, 
                   numProducers * numOperations * sizeof(int), cudaMemcpyDeviceToHost),
        "копирование результатов производителей"
    );
    
    checkCudaError(
        cudaMemcpy(h_consumerResults, d_consumerResults, 
                   numConsumers * numOperations * sizeof(int), cudaMemcpyDeviceToHost),
        "копирование результатов потребителей"
    );
    
    // подсчитываем успешные операции
    int producerSuccess = 0, consumerSuccess = 0;
    for (int i = 0; i < numProducers * numOperations; i++) {
        if (h_producerResults[i] == 1) producerSuccess++;
    }
    for (int i = 0; i < numConsumers * numOperations; i++) {
        if (h_consumerResults[i] != -1) consumerSuccess++;
    }
    
    // вычисляем время выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // выводим результаты
    std::cout << "Успешных enqueue: " << producerSuccess << " из " 
              << numProducers * numOperations << std::endl;
    std::cout << "Успешных dequeue: " << consumerSuccess << " из " 
              << numConsumers * numOperations << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " микросекунд" << std::endl;
    
    // освобождаем память
    cudaFree(d_buffer);
    cudaFree(d_head);
    cudaFree(d_tail);
    cudaFree(d_size);
    cudaFree(d_producerResults);
    cudaFree(d_consumerResults);
    delete[] h_producerResults;
    delete[] h_consumerResults;
}

// функция для тестирования оптимизации с разделяемой памятью
void testSharedMemoryOptimization() {
    std::cout << "\n=== Тестирование оптимизации с разделяемой памятью ===" << std::endl;
    
    const int capacity = 256;           // размер буфера
    const int numThreads = 32;          // количество потоков
    const int numOperations = 8;        // операций на поток
    const int resultsSize = numThreads * numOperations;
    
    // выделяем память на устройстве
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, capacity * sizeof(int)),
        "выделение глобальной памяти"
    );
    
    int *d_results;
    checkCudaError(
        cudaMalloc(&d_results, resultsSize * sizeof(int)),
        "выделение памяти для результатов"
    );
    
    // размер разделяемой памяти в байтах
    int sharedMemSize = capacity * sizeof(int);
    
    // запускаем ядро с замером времени
    auto start = std::chrono::high_resolution_clock::now();
    testSharedMemoryQueue<<<1, numThreads, sharedMemSize>>>(
        d_buffer, capacity, d_results, numOperations
    );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    checkCudaError(cudaGetLastError(), "запуск ядра с разделяемой памятью");
    
    // копируем результаты
    int *h_results = new int[resultsSize];
    checkCudaError(
        cudaMemcpy(h_results, d_results, resultsSize * sizeof(int), cudaMemcpyDeviceToHost),
        "копирование результатов"
    );
    
    // подсчитываем успешные операции
    int successCount = 0;
    for (int i = 0; i < resultsSize; i++) {
        if (h_results[i] != -1) successCount++;
    }
    
    // вычисляем время выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // выводим результаты
    std::cout << "Успешных операций: " << successCount << " из " << resultsSize << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " микросекунд" << std::endl;
    std::cout << "Преимущество: использование быстрой разделяемой памяти вместо глобальной" << std::endl;
    
    // освобождаем память
    cudaFree(d_buffer);
    cudaFree(d_results);
    delete[] h_results;
}

// главная функция программы
int main() {
    std::cout << "Программа тестирования дополнительных заданий" << std::endl;
    
    // тестируем mpmc очередь
    testMPMCQueue();
    
    // тестируем оптимизацию с разделяемой памятью
    testSharedMemoryOptimization();
    
    std::cout << "\nПрограмма завершена успешно!" << std::endl;
    return 0;
}
