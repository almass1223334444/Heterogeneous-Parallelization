#include <stdio.h>
// библиотека cuda для работы с gpu
#include <cuda_runtime.h>

// функция-ядро для сложения векторов
// выполняется на gpu
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // вычисляем индекс элемента
    // который обрабатывает текущий поток
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем не вышли ли за пределы массива
    if (idx < n) {
        // складываем соответствующие элементы
        c[idx] = a[idx] + b[idx];
    }
}

// функция для тестирования конкретной конфигурации
// принимает указатели на данные, размер и параметры запуска
void testConfiguration(float *d_a, float *d_b, float *d_c, int n, 
                       int blockSize, const char* description) {
    // вычисляем сколько блоков понадобится
    // округляем вверх
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // создаем события cuda для замера времени
    cudaEvent_t start, stop;
    // инициализируем событие старта
    cudaEventCreate(&start);
    // инициализируем событие финиша
    cudaEventCreate(&stop);
    
    // фиксируем время начала
    cudaEventRecord(start);
    
    // запускаем ядро с заданной конфигурацией
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // фиксируем время окончания
    cudaEventRecord(stop);
    // ждем пока gpu завершит работу
    cudaEventSynchronize(stop);
    
    // переменная для времени выполнения
    float milliseconds = 0;
    // считаем разницу между событиями
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // переменные для информации об occupancy
    // minGridSize - минимальный размер сетки
    int minGridSize, optimalBlockSize;
    // функция cuda которая подбирает оптимальный размер блока
    // для максимальной загрузки gpu
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                       vectorAdd, 0, 0);
    
    // выводим описание конфигурации
    printf("%s\n", description);
    // выводим размер блока
    printf("  Размер блока: %d\n", blockSize);
    // выводим количество блоков
    printf("  Количество блоков: %d\n", numBlocks);
    // выводим время выполнения
    printf("  Время выполнения: %.3f мс\n", milliseconds);
    // выводим рекомендацию cuda
    printf("  Рекомендуемый размер блока: %d\n\n", optimalBlockSize);
    
    // удаляем событие старта
    cudaEventDestroy(start);
    // удаляем событие финиша
    cudaEventDestroy(stop);
}

// главная функция программы
int main() {
    // размер массивов - миллион элементов
    int n = 1000000;
    // размер в байтах
    size_t size = n * sizeof(float);
    
    // выделяем память на cpu для массива a
    float *h_a = (float*)malloc(size);
    // выделяем память на cpu для массива b
    float *h_b = (float*)malloc(size);
    // выделяем память на cpu для результата c
    float *h_c = (float*)malloc(size);
    
    // заполняем массивы данными
    // проходим по всем элементам
    for (int i = 0; i < n; i++) {
        // массив a - индексы как float
        h_a[i] = i * 1.0f;
        // массив b - удвоенные индексы
        h_b[i] = i * 2.0f;
    }
    
    // объявляем указатели для gpu памяти
    float *d_a, *d_b, *d_c;
    // выделяем память на gpu для a
    cudaMalloc(&d_a, size);
    // выделяем память на gpu для b
    cudaMalloc(&d_b, size);
    // выделяем память на gpu для c
    cudaMalloc(&d_c, size);
    
    // копируем массив a с cpu на gpu
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    // копируем массив b с cpu на gpu
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // выводим заголовок
    printf("Оптимизация конфигурации сетки и блоков\n");
    // выводим размер данных
    printf("Размер массива: %d элементов\n\n", n);
    
    // получаем свойства gpu устройства
    cudaDeviceProp prop;
    // запрашиваем свойства устройства 0
    cudaGetDeviceProperties(&prop, 0);
    // выводим название видеокарты
    printf("Устройство: %s\n", prop.name);
    // выводим максимальное количество потоков в блоке
    printf("Максимальный размер блока: %d\n", prop.maxThreadsPerBlock);
    // выводим размер варпа (обычно 32)
    printf("Warp size: %d\n\n", prop.warpSize);
    
    // тестируем неоптимальную конфигурацию
    // слишком маленький блок - плохо использует ресурсы
    testConfiguration(d_a, d_b, d_c, n, 32, 
                     "Неоптимальная конфигурация (малый размер блока):");
    
    // еще одна плохая конфигурация
    // размер не кратен размеру warp (32)
    testConfiguration(d_a, d_b, d_c, n, 100, 
                     "Неоптимальная конфигурация (не кратен warp):");
    
    // оптимальная конфигурация
    // размер кратен warp и хорошо загружает gpu
    testConfiguration(d_a, d_b, d_c, n, 256, 
                     "Оптимальная конфигурация:");
    
    // переменные для автоматического подбора
    int minGridSize, optimalBlockSize;
    // cuda сама подбирает оптимальный размер
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &optimalBlockSize, 
                                       vectorAdd, 0, 0);
    // тестируем с автоматически подобранным размером
    testConfiguration(d_a, d_b, d_c, n, optimalBlockSize, 
                     "Автоматически оптимизированная конфигурация:");
    
    // копируем результат с gpu на cpu
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // выводим заголовок проверки
    printf("Проверка корректности:\n");
    // флаг корректности результата
    bool correct = true;
    // проверяем все элементы
    for (int i = 0; i < n; i++) {
        // сравниваем результат с ожидаемым
        if (h_c[i] != h_a[i] + h_b[i]) {
            // если не совпадает - ставим флаг
            correct = false;
            // прерываем проверку
            break;
        }
    }
    // выводим результат проверки
    printf("Результат %s\n", correct ? "верный" : "неверный");
    
    // освобождаем память gpu для a
    cudaFree(d_a);
    // освобождаем память gpu для b
    cudaFree(d_b);
    // освобождаем память gpu для c
    cudaFree(d_c);
    // освобождаем память cpu для a
    free(h_a);
    // освобождаем память cpu для b
    free(h_b);
    // освобождаем память cpu для c
    free(h_c);
    
    // возвращаем 0 - программа успешно завершена
    return 0;
}