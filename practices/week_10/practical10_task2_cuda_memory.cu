#include <iostream>
#include <cuda_runtime.h>

// ядро с эффективным (коалесцированным) доступом к памяти
// потоки в одном warp обращаются к последовательным адресам памяти
__global__ void coalescedAccess(float* input, float* output, int size) {
    // вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы массива
    if (idx < size) {
        // последовательный доступ - соседние потоки обращаются к соседним элементам
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// ядро с неэффективным доступом к памяти
// потоки обращаются к памяти с большим шагом (stride)
__global__ void stridedAccess(float* input, float* output, int size) {
    // вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы
    if (idx < size) {
        // доступ с шагом 32 - потоки в одном warp обращаются к далёким адресам
        // это разрушает коалесцированный доступ
        int stridedIdx = (idx * 32) % size;
        output[idx] = input[stridedIdx] * 2.0f + 1.0f;
    }
}

// оптимизированное ядро с использованием разделяемой памяти
__global__ void sharedMemoryKernel(float* input, float* output, int size) {
    // выделяем разделяемую память для блока потоков
    __shared__ float sharedData[256];
    
    // вычисляем индексы
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    
    // загружаем данные из глобальной памяти в разделяемую (коалесцированно)
    if (globalIdx < size) {
        sharedData[localIdx] = input[globalIdx];
    }
    
    // ждём пока все потоки блока загрузят данные
    __syncthreads();
    
    // обрабатываем данные из быстрой разделяемой памяти
    if (globalIdx < size) {
        float value = sharedData[localIdx];
        value = value * 2.0f + 1.0f;
        // записываем результат обратно в глобальную память
        output[globalIdx] = value;
    }
}

// оптимизированное ядро с увеличенной работой на поток
// каждый поток обрабатывает несколько элементов
__global__ void optimizedThreadOrganization(float* input, float* output, int size) {
    // каждый поток обрабатывает 4 элемента
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // обрабатываем 4 элемента подряд
    if (idx < size) {
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
    if (idx + 1 < size) {
        output[idx + 1] = input[idx + 1] * 2.0f + 1.0f;
    }
    if (idx + 2 < size) {
        output[idx + 2] = input[idx + 2] * 2.0f + 1.0f;
    }
    if (idx + 3 < size) {
        output[idx + 3] = input[idx + 3] * 2.0f + 1.0f;
    }
}

// функция для измерения времени выполнения ядра
float measureKernelTime(void (*kernel)(float*, float*, int), 
                        float* d_input, float* d_output, int size,
                        int threadsPerBlock, int numBlocks,
                        const char* kernelName) {
    // создаём события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // засекаем время начала
    cudaEventRecord(start);
    
    // запускаем ядро
    kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);
    
    // засекаем время окончания
    cudaEventRecord(stop);
    
    // ждём завершения
    cudaEventSynchronize(stop);
    
    // вычисляем время в миллисекундах
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результат
    std::cout << kernelName << ": " << milliseconds << " мс\n";
    
    // освобождаем события
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

int main() {
    // размер массива
    const int arraySize = 10000000;  // 10 миллионов элементов
    const int bytes = arraySize * sizeof(float);
    
    std::cout << "Размер массива: " << arraySize << " элементов\n";
    std::cout << "Размер данных: " << bytes / (1024 * 1024) << " МБ\n\n";
    
    // выделяем память на хосте
    float* h_input = new float[arraySize];
    float* h_output = new float[arraySize];
    
    // инициализируем входной массив
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = static_cast<float>(i % 1000) / 10.0f;
    }
    
    // выделяем память на устройстве
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // копируем данные на устройство
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // конфигурация запуска
    int threadsPerBlock = 256;
    int numBlocks = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "=== ТЕСТИРОВАНИЕ ПАТТЕРНОВ ДОСТУПА К ПАМЯТИ ===\n";
    std::cout << "Конфигурация: " << numBlocks << " блоков x " << threadsPerBlock << " потоков\n\n";
    
    // --- тест 1: коалесцированный доступ ---
    float timeCoalesced = measureKernelTime(coalescedAccess, d_input, d_output, 
                                           arraySize, threadsPerBlock, numBlocks,
                                           "1. Коалесцированный доступ");
    
    // --- тест 2: доступ с шагом (stride) ---
    float timeStrided = measureKernelTime(stridedAccess, d_input, d_output, 
                                         arraySize, threadsPerBlock, numBlocks,
                                         "2. Доступ с шагом (неэффективный)");
    
    // --- тест 3: с использованием разделяемой памяти ---
    float timeShared = measureKernelTime(sharedMemoryKernel, d_input, d_output, 
                                        arraySize, threadsPerBlock, numBlocks,
                                        "3. С разделяемой памятью");
    
    // --- тест 4: оптимизированная организация потоков ---
    int optimizedBlocks = (arraySize + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    float timeOptimized = measureKernelTime(optimizedThreadOrganization, d_input, d_output, 
                                           arraySize, threadsPerBlock, optimizedBlocks,
                                           "4. Оптимизированная организация потоков");
    
    // --- анализ результатов ---
    std::cout << "\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===\n";
    std::cout << "Замедление от неэффективного доступа: " 
              << (timeStrided / timeCoalesced) << "x\n";
    std::cout << "Ускорение от разделяемой памяти: " 
              << (timeCoalesced / timeShared) << "x\n";
    std::cout << "Ускорение от оптимизации организации: " 
              << (timeCoalesced / timeOptimized) << "x\n";
    
    // --- детальный анализ ---
    std::cout << "\n=== ДЕТАЛЬНЫЙ АНАЛИЗ ===\n\n";
    
    std::cout << "1. КОАЛЕСЦИРОВАННЫЙ ДОСТУП:\n";
    std::cout << "   + Потоки в warp обращаются к последовательным адресам\n";
    std::cout << "   + GPU объединяет запросы в одну транзакцию памяти\n";
    std::cout << "   + Максимальная пропускная способность памяти\n";
    std::cout << "   Время: " << timeCoalesced << " мс (базовая линия)\n\n";
    
    std::cout << "2. ДОСТУП С ШАГОМ (STRIDE):\n";
    std::cout << "   - Потоки обращаются к памяти с большим шагом\n";
    std::cout << "   - Каждый поток в warp требует отдельную транзакцию\n";
    std::cout << "   - Разрушается коалесцированный доступ\n";
    std::cout << "   - Низкая эффективность использования полосы пропускания\n";
    std::cout << "   Время: " << timeStrided << " мс (медленнее в " 
              << (timeStrided / timeCoalesced) << "x)\n\n";
    
    std::cout << "3. РАЗДЕЛЯЕМАЯ ПАМЯТЬ:\n";
    std::cout << "   + Загрузка данных в быструю on-chip память\n";
    std::cout << "   + Обработка из разделяемой памяти (низкая латентность)\n";
    std::cout << "   + Уменьшение обращений к глобальной памяти\n";
    std::cout << "   - Требует синхронизации потоков\n";
    std::cout << "   Время: " << timeShared << " мс";
    if (timeShared < timeCoalesced) {
        std::cout << " (быстрее в " << (timeCoalesced / timeShared) << "x)\n\n";
    } else {
        std::cout << " (медленнее, накладные расходы)\n\n";
    }
    
    std::cout << "4. ОПТИМИЗИРОВАННАЯ ОРГАНИЗАЦИЯ:\n";
    std::cout << "   + Каждый поток обрабатывает несколько элементов\n";
    std::cout << "   + Уменьшение накладных расходов на запуск потоков\n";
    std::cout << "   + Лучшее использование регистров\n";
    std::cout << "   + Коалесцированный доступ сохранён\n";
    std::cout << "   Время: " << timeOptimized << " мс (ускорение " 
              << (timeCoalesced / timeOptimized) << "x)\n\n";
    
    std::cout << "=== ВЫВОДЫ ===\n";
    std::cout << "1. Паттерн доступа к памяти критически важен для GPU производительности\n";
    std::cout << "2. Неэффективный доступ может замедлить программу в несколько раз\n";
    std::cout << "3. Коалесцированный доступ обеспечивает максимальную пропускную способность\n";
    std::cout << "4. Разделяемая память эффективна для повторного использования данных\n";
    std::cout << "5. Увеличение работы на поток снижает накладные расходы\n";
    std::cout << "6. Правильная организация потоков и доступа - ключ к производительности\n";
    
    // копируем результат для проверки корректности
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // проверка корректности
    std::cout << "\nПроверка корректности (первые 5 элементов):\n";
    for (int i = 0; i < 5; i++) {
        float expected = h_input[i] * 2.0f + 1.0f;
        std::cout << "Input[" << i << "] = " << h_input[i] 
                  << ", Output[" << i << "] = " << h_output[i]
                  << ", Expected = " << expected << "\n";
    }
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
