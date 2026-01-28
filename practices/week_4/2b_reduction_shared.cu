// задание 2b: редукция с использованием разделяемой памяти
// разделяемая память быстрее глобальной и позволяет оптимизировать суммирование

#include <iostream> // для вывода
#include <cuda_runtime.h> // для CUDA
#include <vector> // для массивов
#include <fstream> // для файлов
#include <chrono> // для времени

// kernel с разделяемой памятью - работает намного быстрее
__global__ void reduceSharedMemory(float* input, float* output, int size) {
    // выделяем разделяемую память для блока
    __shared__ float sdata[256]; // разделяемый массив для 256 потоков
    
    // вычисляем индексы
    int tid = threadIdx.x; // локальный индекс потока в блоке
    int i = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    
    // загружаем данные в разделяемую память
    if (i < size) {
        sdata[tid] = input[i]; // копируем из глобальной в разделяемую память
    } else {
        sdata[tid] = 0.0f; // если вышли за границы, записываем 0
    }
    
    // синхронизируем потоки в блоке
    __syncthreads(); // ждем пока все потоки загрузят данные
    
    // редукция внутри блока - суммируем попарно
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { // делим пополам на каждой итерации
        if (tid < s) { // только первая половина потоков работает
            sdata[tid] += sdata[tid + s]; // суммируем пары элементов
        }
        __syncthreads(); // синхронизация после каждого шага
    }
    
    // первый поток блока записывает результат
    if (tid == 0) {
        output[blockIdx.x] = sdata[0]; // записываем сумму блока в глобальную память
    }
}

// функция загрузки данных
void loadData(std::vector<float>& data, int size) {
    std::ifstream inFile("random_data.bin", std::ios::binary); // открываем файл
    if (!inFile) { // проверка
        std::cerr << "Ошибка открытия файла!" << std::endl;
        exit(1); // выход с ошибкой
    }
    data.resize(size); // задаем размер
    inFile.read(reinterpret_cast<char*>(data.data()), size * sizeof(float)); // читаем
    inFile.close(); // закрываем
}

int main() {
    // тестируем на разных размерах
    int sizes[] = {10000, 100000, 1000000}; // массив размеров
    
    // проходим по каждому размеру
    for (int s = 0; s < 3; ++s) {
        int SIZE = sizes[s]; // берем текущий размер
        
        std::cout << "\n=== Тест для размера: " << SIZE << " ===" << std::endl;
        
        // загружаем данные
        std::vector<float> h_data; // массив на CPU
        loadData(h_data, SIZE); // загружаем из файла
        
        // считаем эталонную сумму на CPU
        float cpu_sum = 0.0f; // переменная суммы
        for (int i = 0; i < SIZE; ++i) {
            cpu_sum += h_data[i]; // суммируем элементы
        }
        std::cout << "Эталонная сумма (CPU): " << cpu_sum << std::endl;
        
        // выделяем память на GPU
        float* d_input; // входные данные на GPU
        float* d_output; // выходные данные на GPU
        
        cudaMalloc(&d_input, SIZE * sizeof(float)); // выделяем под входной массив
        
        // вычисляем количество блоков
        int threadsPerBlock = 256; // потоков в блоке (размер разделяемой памяти)
        int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock; // округляем вверх
        
        cudaMalloc(&d_output, blocks * sizeof(float)); // выделяем под результаты блоков
        
        // копируем данные на GPU
        cudaMemcpy(d_input, h_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        // создаем события для времени
        cudaEvent_t start, stop; // события
        cudaEventCreate(&start); // создаем начало
        cudaEventCreate(&stop); // создаем конец
        
        // запускаем первую редукцию
        cudaEventRecord(start); // начинаем отсчет
        
        reduceSharedMemory<<<blocks, threadsPerBlock>>>(d_input, d_output, SIZE); // первый проход
        
        // если блоков больше 1, нужна еще одна редукция
        if (blocks > 1) {
            // создаем промежуточный массив
            float* d_temp; // временный массив
            cudaMalloc(&d_temp, blocks * sizeof(float)); // выделяем память
            cudaMemcpy(d_temp, d_output, blocks * sizeof(float), cudaMemcpyDeviceToDevice); // копируем
            
            // вычисляем новое количество блоков
            int blocks2 = (blocks + threadsPerBlock - 1) / threadsPerBlock; // блоков для второго прохода
            
            // запускаем редукцию результатов
            reduceSharedMemory<<<blocks2, threadsPerBlock>>>(d_temp, d_output, blocks);
            
            cudaFree(d_temp); // освобождаем временную память
        }
        
        cudaEventRecord(stop); // записываем конец
        cudaEventSynchronize(stop); // ждем завершения
        
        // получаем время
        float milliseconds = 0; // переменная времени
        cudaEventElapsedTime(&milliseconds, start, stop); // вычисляем время
        
        // копируем результат на CPU
        std::vector<float> h_output(1); // массив на 1 элемент для финальной суммы
        cudaMemcpy(h_output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost);
        
        float gpu_sum = h_output[0]; // получаем итоговую сумму
        
        // выводим результаты
        std::cout << "Сумма на GPU (разделяемая память): " << gpu_sum << std::endl;
        std::cout << "Время выполнения: " << milliseconds << " мс" << std::endl;
        std::cout << "Разница с CPU: " << std::abs(cpu_sum - gpu_sum) << std::endl;
        
        // освобождаем память
        cudaFree(d_input); // освобождаем входные данные
        cudaFree(d_output); // освобождаем выходные данные
        
        // удаляем события
        cudaEventDestroy(start); // удаляем начало
        cudaEventDestroy(stop); // удаляем конец
    }
    
    return 0; // успешное завершение
}
