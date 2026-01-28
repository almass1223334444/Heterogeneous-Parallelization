#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// структура стека для параллельной работы
struct Stack {
    int *data;      // указатель на массив данных стека
    int top;        // индекс верхнего элемента стека
    int capacity;   // максимальная вместимость стека

    // инициализация стека с буфером и размером
    __device__ void init(int *buffer, int size) {
        data = buffer;      // присваиваем указатель на выделенный буфер
        top = -1;           // изначально стек пуст, поэтому top = -1
        capacity = size;    // сохраняем максимальный размер
    }

    // добавление элемента в стек (thread-safe)
    __device__ bool push(int value) {
        // атомарно увеличиваем top и получаем старое значение
        int pos = atomicAdd(&top, 1);
        
        // проверяем, не вышли ли за границы стека
        if (pos < capacity) {
            data[pos] = value;  // записываем значение в стек
            return true;         // успешное добавление
        }
        return false;  // стек переполнен
    }

    // извлечение элемента из стека (thread-safe)
    __device__ bool pop(int *value) {
        // атомарно уменьшаем top и получаем старое значение
        int pos = atomicSub(&top, 1);
        
        // проверяем, есть ли элементы в стеке
        if (pos >= 0) {
            *value = data[pos];  // читаем значение из стека
            return true;          // успешное извлечение
        }
        return false;  // стек пуст
    }
};

// структура очереди для параллельной работы
struct Queue {
    int *data;      // указатель на массив данных очереди
    int head;       // индекс начала очереди (для чтения)
    int tail;       // индекс конца очереди (для записи)
    int capacity;   // максимальная вместимость очереди

    // инициализация очереди с буфером и размером
    __device__ void init(int *buffer, int size) {
        data = buffer;      // присваиваем указатель на выделенный буфер
        head = 0;           // начало очереди на позиции 0
        tail = 0;           // конец очереди на позиции 0
        capacity = size;    // сохраняем максимальный размер
    }

    // добавление элемента в очередь (thread-safe)
    __device__ bool enqueue(int value) {
        // атомарно увеличиваем tail и получаем старое значение
        int pos = atomicAdd(&tail, 1);
        
        // проверяем, не вышли ли за границы очереди
        if (pos < capacity) {
            data[pos] = value;  // записываем значение в очередь
            return true;         // успешное добавление
        }
        return false;  // очередь переполнена
    }

    // извлечение элемента из очереди (thread-safe)
    __device__ bool dequeue(int *value) {
        // атомарно увеличиваем head и получаем старое значение
        int pos = atomicAdd(&head, 1);
        
        // проверяем, не превышает ли head значение tail
        if (pos < tail) {
            *value = data[pos];  // читаем значение из очереди
            return true;          // успешное извлечение
        }
        return false;  // очередь пуста
    }
};

// ядро для тестирования стека: каждый поток добавляет и извлекает элементы
__global__ void testStackKernel(int *buffer, int bufferSize, int *results, int numOperations) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // создаем локальный объект стека (все потоки работают с одним стеком)
    __shared__ Stack stack;
    
    // только первый поток в блоке инициализирует стек
    if (threadIdx.x == 0) {
        stack.init(buffer, bufferSize);
    }
    // синхронизируем потоки, чтобы все дождались инициализации
    __syncthreads();
    
    // каждый поток пытается добавить свой номер в стек
    for (int i = 0; i < numOperations; i++) {
        int value = tid * 1000 + i;  // уникальное значение для каждого потока
        stack.push(value);            // добавляем в стек
    }
    
    // синхронизируем потоки перед извлечением
    __syncthreads();
    
    // каждый поток пытается извлечь элементы из стека
    for (int i = 0; i < numOperations; i++) {
        int value;
        if (stack.pop(&value)) {      // если извлечение успешно
            results[tid * numOperations + i] = value;  // сохраняем результат
        } else {
            results[tid * numOperations + i] = -1;  // помечаем неудачу
        }
    }
}

// ядро для тестирования очереди: каждый поток добавляет и извлекает элементы
__global__ void testQueueKernel(int *buffer, int bufferSize, int *results, int numOperations) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // создаем локальный объект очереди (все потоки работают с одной очередью)
    __shared__ Queue queue;
    
    // только первый поток в блоке инициализирует очередь
    if (threadIdx.x == 0) {
        queue.init(buffer, bufferSize);
    }
    // синхронизируем потоки, чтобы все дождались инициализации
    __syncthreads();
    
    // каждый поток пытается добавить свой номер в очередь
    for (int i = 0; i < numOperations; i++) {
        int value = tid * 1000 + i;  // уникальное значение для каждого потока
        queue.enqueue(value);         // добавляем в очередь
    }
    
    // синхронизируем потоки перед извлечением
    __syncthreads();
    
    // каждый поток пытается извлечь элементы из очереди
    for (int i = 0; i < numOperations; i++) {
        int value;
        if (queue.dequeue(&value)) {  // если извлечение успешно
            results[tid * numOperations + i] = value;  // сохраняем результат
        } else {
            results[tid * numOperations + i] = -1;  // помечаем неудачу
        }
    }
}

// функция для проверки ошибок CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {  // если произошла ошибка
        std::cerr << "CUDA ошибка: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);  // завершаем программу
    }
}

// функция для тестирования стека
void testStack() {
    std::cout << "\n=== Тестирование стека ===" << std::endl;
    
    const int bufferSize = 1024;     // размер буфера стека
    const int numThreads = 32;       // количество потоков
    const int numOperations = 10;    // операций на поток
    const int resultsSize = numThreads * numOperations;  // размер массива результатов
    
    // выделяем память на устройстве для буфера стека
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, bufferSize * sizeof(int)),
        "выделение памяти для буфера стека"
    );
    
    // выделяем память на устройстве для результатов
    int *d_results;
    checkCudaError(
        cudaMalloc(&d_results, resultsSize * sizeof(int)),
        "выделение памяти для результатов стека"
    );
    
    // запускаем ядро с замером времени
    auto start = std::chrono::high_resolution_clock::now();
    testStackKernel<<<1, numThreads>>>(d_buffer, bufferSize, d_results, numOperations);
    cudaDeviceSynchronize();  // ждем завершения ядра
    auto end = std::chrono::high_resolution_clock::now();
    
    // проверяем ошибки после запуска ядра
    checkCudaError(cudaGetLastError(), "запуск ядра стека");
    
    // выделяем память на хосте для результатов
    int *h_results = new int[resultsSize];
    
    // копируем результаты с устройства на хост
    checkCudaError(
        cudaMemcpy(h_results, d_results, resultsSize * sizeof(int), cudaMemcpyDeviceToHost),
        "копирование результатов стека"
    );
    
    // подсчитываем успешные операции
    int successCount = 0;
    for (int i = 0; i < resultsSize; i++) {
        if (h_results[i] != -1) {  // если операция была успешной
            successCount++;
        }
    }
    
    // вычисляем время выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // выводим результаты
    std::cout << "Успешных операций pop: " << successCount << " из " << resultsSize << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " микросекунд" << std::endl;
    
    // освобождаем память
    cudaFree(d_buffer);
    cudaFree(d_results);
    delete[] h_results;
}

// функция для тестирования очереди
void testQueue() {
    std::cout << "\n=== Тестирование очереди ===" << std::endl;
    
    const int bufferSize = 1024;     // размер буфера очереди
    const int numThreads = 32;       // количество потоков
    const int numOperations = 10;    // операций на поток
    const int resultsSize = numThreads * numOperations;  // размер массива результатов
    
    // выделяем память на устройстве для буфера очереди
    int *d_buffer;
    checkCudaError(
        cudaMalloc(&d_buffer, bufferSize * sizeof(int)),
        "выделение памяти для буфера очереди"
    );
    
    // выделяем память на устройстве для результатов
    int *d_results;
    checkCudaError(
        cudaMalloc(&d_results, resultsSize * sizeof(int)),
        "выделение памяти для результатов очереди"
    );
    
    // запускаем ядро с замером времени
    auto start = std::chrono::high_resolution_clock::now();
    testQueueKernel<<<1, numThreads>>>(d_buffer, bufferSize, d_results, numOperations);
    cudaDeviceSynchronize();  // ждем завершения ядра
    auto end = std::chrono::high_resolution_clock::now();
    
    // проверяем ошибки после запуска ядра
    checkCudaError(cudaGetLastError(), "запуск ядра очереди");
    
    // выделяем память на хосте для результатов
    int *h_results = new int[resultsSize];
    
    // копируем результаты с устройства на хост
    checkCudaError(
        cudaMemcpy(h_results, d_results, resultsSize * sizeof(int), cudaMemcpyDeviceToHost),
        "копирование результатов очереди"
    );
    
    // подсчитываем успешные операции
    int successCount = 0;
    for (int i = 0; i < resultsSize; i++) {
        if (h_results[i] != -1) {  // если операция была успешной
            successCount++;
        }
    }
    
    // вычисляем время выполнения
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // выводим результаты
    std::cout << "Успешных операций dequeue: " << successCount << " из " << resultsSize << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " микросекунд" << std::endl;
    
    // освобождаем память
    cudaFree(d_buffer);
    cudaFree(d_results);
    delete[] h_results;
}

// главная функция программы
int main() {
    std::cout << "Программа тестирования параллельных структур данных на CUDA" << std::endl;
    
    // тестируем стек
    testStack();
    
    // тестируем очередь
    testQueue();
    
    std::cout << "\nПрограмма завершена успешно!" << std::endl;
    return 0;
}
