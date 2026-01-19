#include <stdio.h>
// подключаем библиотеку cuda runtime для работы с gpu
#include <cuda_runtime.h>

// объявляем функцию-ядро которая выполняется на gpu
// __global__ означает что функция вызывается с cpu но работает на gpu
// принимает указатель на массив, множитель и размер массива
__global__ void multiplyGlobal(float *arr, float multiplier, int n) {
    // считаем глобальный индекс текущего потока
    // blockIdx.x - номер текущего блока
    // blockDim.x - количество потоков в блоке
    // threadIdx.x - номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем не вышли ли мы за границы массива
    if (idx < n) {
        // умножаем элемент массива на множитель
        // arr[idx] - обращение к глобальной памяти gpu
        arr[idx] *= multiplier;
    }
    // конец функции-ядра
}

// объявляем второе ядро с использованием разделяемой памяти
__global__ void multiplyShared(float *arr, float multiplier, int n) {
    // объявляем указатель на разделяемую память
    // extern означает что размер указывается при запуске ядра
    // __shared__ означает что память доступна всем потокам блока
    extern __shared__ float shared_data[];
    
    // считаем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы массива
    if (idx < n) {
        // копируем данные из глобальной памяти в разделяемую
        // threadIdx.x - локальный индекс для разделяемой памяти
        shared_data[threadIdx.x] = arr[idx];
    }
    
    // синхронизируем все потоки в блоке
    // ждем пока все потоки скопируют свои данные
    __syncthreads();
    
    // выполняем вычисление в разделяемой памяти
    if (idx < n) {
        // умножаем элемент в разделяемой памяти
        shared_data[threadIdx.x] *= multiplier;
    }
    
    // снова синхронизируем потоки
    // ждем завершения вычислений
    __syncthreads();
    
    // копируем результат обратно в глобальную память
    if (idx < n) {
        // записываем из разделяемой в глобальную память
        arr[idx] = shared_data[threadIdx.x];
    }
    // конец второго ядра
}

// главная функция программы
int main() {
    // объявляем размер массива
    int n = 1000000;
    // вычисляем размер в байтах
    // sizeof(float) = 4 байта
    size_t size = n * sizeof(float);
    
    // выделяем память на cpu (host)
    // malloc возвращает указатель на выделенную память
    float *h_arr = (float*)malloc(size);
    
    // инициализируем массив значениями
    // цикл от 0 до n-1
    for (int i = 0; i < n; i++) {
        // присваиваем каждому элементу его индекс в виде float
        h_arr[i] = i * 1.0f;
    }
    
    // задаем множитель
    float multiplier = 2.5f;
    
    // объявляем указатель на память gpu
    float *d_arr;
    // выделяем память на gpu (device)
    // cudaMalloc принимает адрес указателя и размер
    cudaMalloc(&d_arr, size);
    
    // задаем размер блока потоков
    int blockSize = 256;
    // вычисляем количество блоков
    // округляем вверх деля с остатком
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // выводим информацию о конфигурации
    printf("Тестирование с массивом из %d элементов\n", n);
    printf("Блоков: %d, потоков на блок: %d\n\n", numBlocks, blockSize);
    
    // копируем данные с cpu на gpu
    // cudaMemcpyHostToDevice означает направление копирования
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    
    // создаем события cuda для замера времени
    // события это метки времени на gpu
    cudaEvent_t start, stop;
    // инициализируем событие начала
    cudaEventCreate(&start);
    // инициализируем событие конца
    cudaEventCreate(&stop);
    
    // записываем событие начала выполнения
    cudaEventRecord(start);
    
    // запускаем ядро с глобальной памятью
    // <<<numBlocks, blockSize>>> - конфигурация запуска
    multiplyGlobal<<<numBlocks, blockSize>>>(d_arr, multiplier, n);
    
    // записываем событие окончания
    cudaEventRecord(stop);
    // ждем завершения всех операций на gpu
    cudaEventSynchronize(stop);
    
    // вычисляем время выполнения в миллисекундах
    float milliseconds = 0;
    // cudaEventElapsedTime записывает разницу во времени
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результат для глобальной памяти
    printf("Время с глобальной памятью: %.3f мс\n", milliseconds);
    
    // снова копируем исходные данные на gpu
    // чтобы тест был честным
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    
    // записываем начало второго теста
    cudaEventRecord(start);
    
    // запускаем ядро с разделяемой памятью
    // третий параметр - размер разделяемой памяти в байтах
    multiplyShared<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_arr, multiplier, n);
    
    // записываем конец второго теста
    cudaEventRecord(stop);
    // ждем завершения
    cudaEventSynchronize(stop);
    
    // вычисляем время для разделяемой памяти
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // выводим результат
    printf("Время с разделяемой памятью: %.3f мс\n", milliseconds);
    
    // копируем результат обратно на cpu
    // cudaMemcpyDeviceToHost - с gpu на cpu
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    
    // освобождаем память gpu
    cudaFree(d_arr);
    // уничтожаем событие начала
    cudaEventDestroy(start);
    // уничтожаем событие конца
    cudaEventDestroy(stop);
    // освобождаем память cpu
    free(h_arr);
    
    // возвращаем 0 - программа завершилась успешно
    return 0;
}