#include <stdio.h>
// подключаем cuda runtime
#include <cuda_runtime.h>

// ядро с коалесцированным доступом
// потоки читают последовательные элементы памяти
__global__ void processCoalesced(float *arr, int n) {
    // вычисляем индекс текущего потока
    // последовательный доступ - соседние потоки берут соседние элементы
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем не вышли ли за границы
    if (idx < n) {
        // читаем элемент из глобальной памяти
        // соседние потоки читают соседние адреса - это эффективно
        float value = arr[idx];
        // выполняем простое вычисление
        // умножаем на 2 и прибавляем 1
        value = value * 2.0f + 1.0f;
        // записываем обратно по тому же адресу
        // запись тоже коалесцированная
        arr[idx] = value;
    }
}

// ядро с некоалесцированным доступом
// потоки читают разбросанные элементы памяти
__global__ void processUncoalesced(float *arr, int n) {
    // вычисляем базовый индекс
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // задаем шаг для некоалесцированного доступа
    // большой шаг означает что соседние потоки читают далекие элементы
    int stride = 32;
    // вычисляем некоалесцированный индекс
    // соседние потоки будут обращаться к элементам далеко друг от друга
    // это неэффективно для памяти gpu
    int uncoalesced_idx = (idx / stride) * stride * stride + (idx % stride);
    
    // проверяем границы с новым индексом
    if (uncoalesced_idx < n) {
        // читаем элемент с разбросанным адресом
        // gpu не может объединить эти обращения в одну транзакцию
        float value = arr[uncoalesced_idx];
        // выполняем то же самое вычисление
        value = value * 2.0f + 1.0f;
        // записываем обратно
        // запись тоже некоалесцированная
        arr[uncoalesced_idx] = value;
    }
}

// главная функция
int main() {
    // размер массива - миллион элементов
    int n = 1000000;
    // размер в байтах
    size_t size = n * sizeof(float);
    
    // выделяем память на cpu
    float *h_arr = (float*)malloc(size);
    
    // заполняем массив
    // цикл от 0 до n-1
    for (int i = 0; i < n; i++) {
        // каждый элемент равен своему индексу
        h_arr[i] = i * 1.0f;
    }
    
    // указатель на память gpu
    float *d_arr;
    // выделяем память на gpu
    cudaMalloc(&d_arr, size);
    
    // размер блока потоков
    int blockSize = 256;
    // количество блоков
    // округляем вверх
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // выводим информацию о тесте
    printf("Сравнение коалесцированного и некоалесцированного доступа\n");
    printf("Размер массива: %d элементов\n", n);
    printf("Конфигурация: %d блоков по %d потоков\n\n", numBlocks, blockSize);
    
    // создаем события для замера времени
    cudaEvent_t start, stop;
    // инициализируем событие начала
    cudaEventCreate(&start);
    // инициализируем событие конца
    cudaEventCreate(&stop);
    
    // первый тест - коалесцированный доступ
    // копируем данные на gpu
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    
    // отмечаем начало выполнения
    cudaEventRecord(start);
    
    // запускаем ядро с коалесцированным доступом
    processCoalesced<<<numBlocks, blockSize>>>(d_arr, n);
    
    // отмечаем конец выполнения
    cudaEventRecord(stop);
    // ждем завершения всех операций
    cudaEventSynchronize(stop);
    
    // переменная для времени
    float milliseconds = 0;
    // вычисляем время выполнения
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результат
    printf("Коалесцированный доступ:     %.3f мс\n", milliseconds);
    
    // второй тест - некоалесцированный доступ
    // снова копируем исходные данные
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    
    // отмечаем начало
    cudaEventRecord(start);
    
    // запускаем ядро с некоалесцированным доступом
    processUncoalesced<<<numBlocks, blockSize>>>(d_arr, n);
    
    // отмечаем конец
    cudaEventRecord(stop);
    // ждем завершения
    cudaEventSynchronize(stop);
    
    // вычисляем время
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результат
    printf("Некоалесцированный доступ:   %.3f мс\n", milliseconds);
    
    // освобождаем память gpu
    cudaFree(d_arr);
    // уничтожаем событие начала
    cudaEventDestroy(start);
    // уничтожаем событие конца
    cudaEventDestroy(stop);
    // освобождаем память cpu
    free(h_arr);
    
    // программа завершилась успешно
    return 0;
}