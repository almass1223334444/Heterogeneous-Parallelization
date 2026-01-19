#include <stdio.h>
// подключаем cuda runtime для работы с видеокартой
#include <cuda_runtime.h>

// функция-ядро для сложения двух массивов
// __global__ значит функция запускается на gpu
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // вычисляем индекс элемента для текущего потока
    // blockIdx.x - номер блока
    // blockDim.x - размер блока
    // threadIdx.x - номер потока в блоке
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем не вышли ли за границы массива
    if (idx < n) {
        // складываем элементы двух массивов
        // записываем результат в третий массив
        c[idx] = a[idx] + b[idx];
    }
}

// функция для тестирования с конкретным размером блока
// принимает указатели на массивы, размер массива и размер блока
void testBlockSize(float *d_a, float *d_b, float *d_c, int n, int blockSize) {
    // вычисляем сколько блоков нужно
    // округляем вверх через деление с остатком
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // создаем событие для начала отсчета
    cudaEvent_t start, stop;
    // инициализируем событие start
    cudaEventCreate(&start);
    // инициализируем событие stop
    cudaEventCreate(&stop);
    
    // отмечаем время начала на gpu
    cudaEventRecord(start);
    
    // запускаем ядро сложения векторов
    // передаем конфигурацию и параметры
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // отмечаем время окончания
    cudaEventRecord(stop);
    // ждем пока gpu завершит все операции
    cudaEventSynchronize(stop);
    
    // переменная для хранения времени в миллисекундах
    float milliseconds = 0;
    // вычисляем прошедшее время между событиями
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результаты теста
    // %4d - число шириной 4 символа
    printf("Размер блока: %4d | Блоков: %5d | Время: %.3f мс\n", 
           blockSize, numBlocks, milliseconds);
    
    // удаляем событие start
    cudaEventDestroy(start);
    // удаляем событие stop
    cudaEventDestroy(stop);
}

// главная функция программы
int main() {
    // задаем размер массивов
    int n = 1000000;
    // вычисляем размер в байтах
    size_t size = n * sizeof(float);
    
    // выделяем память на cpu для первого массива
    float *h_a = (float*)malloc(size);
    // выделяем память на cpu для второго массива
    float *h_b = (float*)malloc(size);
    // выделяем память на cpu для результата
    float *h_c = (float*)malloc(size);
    
    // заполняем массивы данными
    // цикл по всем элементам
    for (int i = 0; i < n; i++) {
        // первый массив - просто индексы
        h_a[i] = i * 1.0f;
        // второй массив - удвоенные индексы
        h_b[i] = i * 2.0f;
    }
    
    // объявляем указатели для памяти на gpu
    float *d_a, *d_b, *d_c;
    // выделяем память на gpu для первого массива
    cudaMalloc(&d_a, size);
    // выделяем память на gpu для второго массива
    cudaMalloc(&d_b, size);
    // выделяем память на gpu для результата
    cudaMalloc(&d_c, size);
    
    // копируем первый массив с cpu на gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    // копируем второй массив с cpu на gpu
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // выводим заголовок
    printf("Исследование влияния размера блока\n");
    // выводим размер массива
    printf("Размер массивов: %d элементов\n\n", n);
    
    // создаем массив с размерами блоков для тестирования
    int blockSizes[] = {64, 128, 256, 512, 1024};
    // количество тестов
    int numTests = 5;
    
    // цикл по всем размерам блоков
    for (int i = 0; i < numTests; i++) {
        // вызываем функцию тестирования с текущим размером
        testBlockSize(d_a, d_b, d_c, n, blockSizes[i]);
    }
    
    // копируем результат с gpu обратно на cpu
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // выводим заголовок для проверки
    printf("\nПроверка корректности (первые 5 элементов):\n");
    // проверяем первые 5 элементов
    for (int i = 0; i < 5; i++) {
        // выводим пример сложения
        // %.1f - число с одним знаком после запятой
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // освобождаем память gpu для первого массива
    cudaFree(d_a);
    // освобождаем память gpu для второго массива
    cudaFree(d_b);
    // освобождаем память gpu для результата
    cudaFree(d_c);
    // освобождаем память cpu для первого массива
    free(h_a);
    // освобождаем память cpu для второго массива
    free(h_b);
    // освобождаем память cpu для результата
    free(h_c);
    
    // программа завершилась успешно
    return 0;
}