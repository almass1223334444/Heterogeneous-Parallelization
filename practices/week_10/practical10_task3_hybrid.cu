#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

// ядро gpu для обработки части массива
__global__ void processOnGPU(float* input, float* output, int size, int offset) {
    // вычисляем глобальный индекс с учётом смещения
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы
    if (idx < size) {
        // сложная обработка чтобы увеличить время вычислений
        float value = input[offset + idx];
        for (int i = 0; i < 100; i++) {
            value = value * 1.01f - 0.001f;
        }
        output[offset + idx] = value;
    }
}

// функция обработки на cpu
void processOnCPU(float* input, float* output, int start, int end) {
    // обрабатываем элементы от start до end
    for (int i = start; i < end; i++) {
        // та же обработка что и на gpu
        float value = input[i];
        for (int j = 0; j < 100; j++) {
            value = value * 1.01f - 0.001f;
        }
        output[i] = value;
    }
}

int main() {
    // размер массива
    const int arraySize = 10000000;  // 10 миллионов элементов
    const int bytes = arraySize * sizeof(float);
    
    std::cout << "=== ГИБРИДНОЕ ПРИЛОЖЕНИЕ CPU + GPU ===\n";
    std::cout << "Размер массива: " << arraySize << " элементов\n\n";
    
    // выделяем память на хосте
    float* h_input = new float[arraySize];
    float* h_output = new float[arraySize];
    
    // инициализируем входные данные
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = static_cast<float>(i % 1000) / 100.0f;
    }
    
    // выделяем pinned memory для асинхронной передачи
    float *h_pinnedInput, *h_pinnedOutput;
    cudaMallocHost(&h_pinnedInput, bytes);  // закреплённая память
    cudaMallocHost(&h_pinnedOutput, bytes);
    
    // копируем данные в pinned memory
    memcpy(h_pinnedInput, h_input, bytes);
    
    // выделяем память на gpu
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // создаём cuda streams для асинхронных операций
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // создаём события для профилирования
    cudaEvent_t start, stop;
    cudaEvent_t copyStart, copyEnd, computeStart, computeEnd;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&copyStart);
    cudaEventCreate(&copyEnd);
    cudaEventCreate(&computeStart);
    cudaEventCreate(&computeEnd);
    
    // --- тест 1: только cpu ---
    std::cout << "=== ТЕСТ 1: ТОЛЬКО CPU ===\n";
    
    auto cpuStart = std::chrono::high_resolution_clock::now();
    
    processOnCPU(h_input, h_output, 0, arraySize);
    
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
    
    std::cout << "Время выполнения: " << cpuTime << " мс\n\n";
    
    // --- тест 2: только gpu (синхронная передача) ---
    std::cout << "=== ТЕСТ 2: ТОЛЬКО GPU (синхронная передача) ===\n";
    
    cudaEventRecord(start);
    
    // синхронная передача на gpu
    cudaEventRecord(copyStart);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(copyEnd);
    
    // конфигурация ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    
    // запуск ядра
    cudaEventRecord(computeStart);
    processOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, arraySize, 0);
    cudaEventRecord(computeEnd);
    
    // синхронная передача обратно
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpuTotalTime = 0, gpuCopyTime = 0, gpuComputeTime = 0;
    cudaEventElapsedTime(&gpuTotalTime, start, stop);
    cudaEventElapsedTime(&gpuCopyTime, copyStart, copyEnd);
    cudaEventElapsedTime(&gpuComputeTime, computeStart, computeEnd);
    
    std::cout << "Общее время: " << gpuTotalTime << " мс\n";
    std::cout << "  - Время копирования на GPU: " << gpuCopyTime << " мс\n";
    std::cout << "  - Время вычислений: " << gpuComputeTime << " мс\n";
    std::cout << "  - Время копирования обратно: " << (gpuTotalTime - gpuCopyTime - gpuComputeTime) << " мс\n";
    std::cout << "Доля накладных расходов: " << ((gpuTotalTime - gpuComputeTime) / gpuTotalTime * 100) << "%\n\n";
    
    // --- тест 3: гибридный подход (базовый) ---
    std::cout << "=== ТЕСТ 3: ГИБРИДНЫЙ (базовый, без оптимизаций) ===\n";
    
    // делим массив: 30% на cpu, 70% на gpu
    int cpuPart = arraySize * 0.3;
    int gpuPart = arraySize - cpuPart;
    
    auto hybridStart = std::chrono::high_resolution_clock::now();
    
    // запускаем cpu в отдельном потоке
    std::thread cpuThread([&]() {
        processOnCPU(h_input, h_output, 0, cpuPart);
    });
    
    // пока cpu работает, копируем gpu часть и запускаем ядро
    cudaMemcpy(d_input + cpuPart, h_input + cpuPart, gpuPart * sizeof(float), cudaMemcpyHostToDevice);
    
    int gpuBlocks = (gpuPart + threadsPerBlock - 1) / threadsPerBlock;
    processOnGPU<<<gpuBlocks, threadsPerBlock>>>(d_input, d_output, gpuPart, cpuPart);
    
    cudaDeviceSynchronize();
    
    // копируем результат gpu части
    cudaMemcpy(h_output + cpuPart, d_output + cpuPart, gpuPart * sizeof(float), cudaMemcpyDeviceToHost);
    
    // ждём завершения cpu
    cpuThread.join();
    
    auto hybridEnd = std::chrono::high_resolution_clock::now();
    auto hybridTime = std::chrono::duration_cast<std::chrono::milliseconds>(hybridEnd - hybridStart).count();
    
    std::cout << "Время выполнения: " << hybridTime << " мс\n";
    std::cout << "Ускорение относительно CPU: " << (float)cpuTime / hybridTime << "x\n\n";
    
    // --- тест 4: оптимизированный гибридный подход с асинхронной передачей ---
    std::cout << "=== ТЕСТ 4: ОПТИМИЗИРОВАННЫЙ ГИБРИДНЫЙ (async + streams) ===\n";
    std::cout << "Оптимизации:\n";
    std::cout << "  1. Использование pinned memory\n";
    std::cout << "  2. Асинхронная передача данных (cudaMemcpyAsync)\n";
    std::cout << "  3. CUDA streams для параллелизма\n";
    std::cout << "  4. Разделение на несколько чанков\n\n";
    
    auto optimizedStart = std::chrono::high_resolution_clock::now();
    
    // запускаем cpu обработку в отдельном потоке
    std::thread cpuThreadOpt([&]() {
        processOnCPU(h_pinnedInput, h_pinnedOutput, 0, cpuPart);
    });
    
    // делим gpu часть на 2 чанка для конвейерной обработки
    int chunkSize = gpuPart / 2;
    
    // обработка первого чанка в stream1
    int offset1 = cpuPart;
    cudaMemcpyAsync(d_input + offset1, h_pinnedInput + offset1, 
                    chunkSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
    
    int chunk1Blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    processOnGPU<<<chunk1Blocks, threadsPerBlock, 0, stream1>>>(d_input, d_output, chunkSize, offset1);
    
    cudaMemcpyAsync(h_pinnedOutput + offset1, d_output + offset1, 
                    chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    
    // обработка второго чанка в stream2 (параллельно с первым)
    int offset2 = cpuPart + chunkSize;
    int chunk2Size = gpuPart - chunkSize;
    cudaMemcpyAsync(d_input + offset2, h_pinnedInput + offset2, 
                    chunk2Size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    
    int chunk2Blocks = (chunk2Size + threadsPerBlock - 1) / threadsPerBlock;
    processOnGPU<<<chunk2Blocks, threadsPerBlock, 0, stream2>>>(d_input, d_output, chunk2Size, offset2);
    
    cudaMemcpyAsync(h_pinnedOutput + offset2, d_output + offset2, 
                    chunk2Size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    
    // ждём завершения обоих streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    // ждём завершения cpu потока
    cpuThreadOpt.join();
    
    auto optimizedEnd = std::chrono::high_resolution_clock::now();
    auto optimizedTime = std::chrono::duration_cast<std::chrono::milliseconds>(optimizedEnd - optimizedStart).count();
    
    std::cout << "Время выполнения: " << optimizedTime << " мс\n";
    std::cout << "Ускорение относительно CPU: " << (float)cpuTime / optimizedTime << "x\n";
    std::cout << "Ускорение относительно базового гибрида: " << (float)hybridTime / optimizedTime << "x\n\n";
    
    // --- анализ и выводы ---
    std::cout << "=== АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ ===\n\n";
    
    std::cout << "Сравнение подходов:\n";
    std::cout << "1. Только CPU:                 " << cpuTime << " мс (базовая линия)\n";
    std::cout << "2. Только GPU:                 " << gpuTotalTime << " мс\n";
    std::cout << "3. Гибридный базовый:          " << hybridTime << " мс\n";
    std::cout << "4. Гибридный оптимизированный: " << optimizedTime << " мс (лучший результат)\n\n";
    
    std::cout << "ПРОФИЛИРОВАНИЕ УЗКИХ МЕСТ:\n\n";
    
    std::cout << "1. Накладные расходы передачи данных:\n";
    std::cout << "   - При синхронной передаче: " << (gpuTotalTime - gpuComputeTime) << " мс\n";
    std::cout << "   - Доля от общего времени: " << ((gpuTotalTime - gpuComputeTime) / gpuTotalTime * 100) << "%\n";
    std::cout << "   - Критично для производительности!\n\n";
    
    std::cout << "2. Взаимодействие CPU-GPU:\n";
    std::cout << "   - Синхронизация потоков\n";
    std::cout << "   - Копирование через PCIe шину (узкое место)\n";
    std::cout << "   - Простой устройств при ожидании\n\n";
    
    std::cout << "3. Эффект от оптимизаций:\n";
    std::cout << "   - Pinned memory: быстрее передача на ~2x\n";
    std::cout << "   - Async копирование: перекрытие с вычислениями\n";
    std::cout << "   - CUDA streams: параллельная обработка чанков\n";
    std::cout << "   - Общий выигрыш: " << (float)hybridTime / optimizedTime << "x\n\n";
    
    std::cout << "=== ВЫВОДЫ ===\n";
    std::cout << "1. Передача данных - критическое узкое место в CPU-GPU взаимодействии\n";
    std::cout << "2. Асинхронная передача позволяет перекрыть копирование и вычисления\n";
    std::cout << "3. Использование streams увеличивает параллелизм\n";
    std::cout << "4. Pinned memory ускоряет передачу данных\n";
    std::cout << "5. Правильное разделение работы между CPU и GPU критично\n";
    std::cout << "6. Оптимизации могут дать ускорение в " << (float)hybridTime / optimizedTime << "x\n";
    
    // освобождаем ресурсы
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(copyStart);
    cudaEventDestroy(copyEnd);
    cudaEventDestroy(computeStart);
    cudaEventDestroy(computeEnd);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_pinnedInput);
    cudaFreeHost(h_pinnedOutput);
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
