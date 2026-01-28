// задание 4: полное сравнение производительности всех методов
// программа тестирует все варианты и записывает результаты для графиков

#include <iostream> // для вывода
#include <cuda_runtime.h> // для CUDA
#include <vector> // для массивов
#include <fstream> // для файлов
#include <iomanip> // для форматирования вывода

// kernel с глобальной памятью
__global__ void reduceGlobalMemory(float* input, float* output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    int stride = blockDim.x * gridDim.x; // общее количество потоков
    
    float sum = 0.0f; // локальная сумма
    for (int i = tid; i < size; i += stride) {
        sum += input[i]; // суммируем элементы
    }
    
    output[tid] = sum; // записываем результат
}

// kernel с разделяемой памятью
__global__ void reduceSharedMemory(float* input, float* output, int size) {
    __shared__ float sdata[256]; // разделяемая память
    
    int tid = threadIdx.x; // локальный индекс
    int i = blockIdx.x * blockDim.x + threadIdx.x; // глобальный индекс
    
    // загрузка в разделяемую память
    if (i < size) {
        sdata[tid] = input[i]; // загружаем
    } else {
        sdata[tid] = 0.0f; // заполняем нулями
    }
    
    __syncthreads(); // синхронизация
    
    // редукция
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // суммируем
        }
        __syncthreads(); // синхронизация
    }
    
    // запись результата
    if (tid == 0) {
        output[blockIdx.x] = sdata[0]; // первый поток записывает
    }
}

// функция загрузки данных
void loadData(std::vector<float>& data, int size) {
    std::ifstream inFile("random_data.bin", std::ios::binary); // открываем
    if (!inFile) { // проверка
        std::cerr << "Ошибка: файл не найден! Сначала запустите программу генерации." << std::endl;
        exit(1); // выход
    }
    data.resize(size); // размер
    inFile.read(reinterpret_cast<char*>(data.data()), size * sizeof(float)); // читаем
    inFile.close(); // закрываем
}

// функция для теста редукции с глобальной памятью
float testGlobalMemory(float* d_input, int SIZE) {
    float* d_output; // выходной массив
    
    cudaMalloc(&d_output, 1024 * sizeof(float)); // выделяем память
    
    int threadsPerBlock = 256; // потоков в блоке
    int blocks = 256; // блоков
    
    // события для измерения времени
    cudaEvent_t start, stop; // события
    cudaEventCreate(&start); // создаем
    cudaEventCreate(&stop); // создаем
    
    // запуск
    cudaEventRecord(start); // начало
    reduceGlobalMemory<<<blocks, threadsPerBlock>>>(d_input, d_output, SIZE); // kernel
    cudaEventRecord(stop); // конец
    cudaEventSynchronize(stop); // ждем
    
    // получаем время
    float milliseconds = 0; // переменная
    cudaEventElapsedTime(&milliseconds, start, stop); // вычисляем
    
    // освобождаем память
    cudaFree(d_output); // освобождаем
    cudaEventDestroy(start); // удаляем
    cudaEventDestroy(stop); // удаляем
    
    return milliseconds; // возвращаем время
}

// функция для теста редукции с разделяемой памятью
float testSharedMemory(float* d_input, int SIZE) {
    float* d_output; // выходной массив
    
    int threadsPerBlock = 256; // потоков в блоке
    int blocks = (SIZE + threadsPerBlock - 1) / threadsPerBlock; // блоков
    
    cudaMalloc(&d_output, blocks * sizeof(float)); // выделяем
    
    // события
    cudaEvent_t start, stop; // события
    cudaEventCreate(&start); // создаем
    cudaEventCreate(&stop); // создаем
    
    // запуск
    cudaEventRecord(start); // начало
    reduceSharedMemory<<<blocks, threadsPerBlock>>>(d_input, d_output, SIZE); // kernel
    
    // вторая редукция если нужна
    if (blocks > 1) {
        float* d_temp; // временный массив
        cudaMalloc(&d_temp, blocks * sizeof(float)); // выделяем
        cudaMemcpy(d_temp, d_output, blocks * sizeof(float), cudaMemcpyDeviceToDevice); // копируем
        
        int blocks2 = (blocks + threadsPerBlock - 1) / threadsPerBlock; // новые блоки
        reduceSharedMemory<<<blocks2, threadsPerBlock>>>(d_temp, d_output, blocks); // второй проход
        
        cudaFree(d_temp); // освобождаем
    }
    
    cudaEventRecord(stop); // конец
    cudaEventSynchronize(stop); // ждем
    
    // время
    float milliseconds = 0; // переменная
    cudaEventElapsedTime(&milliseconds, start, stop); // вычисляем
    
    // освобождаем
    cudaFree(d_output); // освобождаем
    cudaEventDestroy(start); // удаляем
    cudaEventDestroy(stop); // удаляем
    
    return milliseconds; // возвращаем
}

int main() {
    // размеры для тестирования
    int sizes[] = {10000, 100000, 1000000}; // три размера
    
    // создаем файл для результатов
    std::ofstream resultsFile("performance_results.csv"); // CSV файл
    resultsFile << "Размер массива,Глобальная память (мс),Разделяемая память (мс),Ускорение" << std::endl;
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ПОЛНОЕ ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // проходим по всем размерам
    for (int s = 0; s < 3; ++s) {
        int SIZE = sizes[s]; // текущий размер
        
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Размер массива: " << SIZE << " элементов" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // загружаем данные
        std::vector<float> h_data; // массив на CPU
        loadData(h_data, SIZE); // загружаем
        
        // выделяем память на GPU
        float* d_input; // входной массив
        cudaMalloc(&d_input, SIZE * sizeof(float)); // выделяем
        cudaMemcpy(d_input, h_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice); // копируем
        
        // тестируем глобальную память
        std::cout << "\n1. Тест редукции с ГЛОБАЛЬНОЙ памятью..." << std::endl;
        float globalTime = testGlobalMemory(d_input, SIZE); // тестируем
        std::cout << "   Время выполнения: " << std::fixed << std::setprecision(3) 
                  << globalTime << " мс" << std::endl;
        
        // тестируем разделяемую память
        std::cout << "\n2. Тест редукции с РАЗДЕЛЯЕМОЙ памятью..." << std::endl;
        float sharedTime = testSharedMemory(d_input, SIZE); // тестируем
        std::cout << "   Время выполнения: " << std::fixed << std::setprecision(3) 
                  << sharedTime << " мс" << std::endl;
        
        // вычисляем ускорение
        float speedup = globalTime / sharedTime; // коэффициент ускорения
        
        std::cout << "\n3. РЕЗУЛЬТАТЫ СРАВНЕНИЯ:" << std::endl;
        std::cout << "   Глобальная память:   " << std::setw(10) << globalTime << " мс" << std::endl;
        std::cout << "   Разделяемая память:  " << std::setw(10) << sharedTime << " мс" << std::endl;
        std::cout << "   Ускорение:           " << std::setw(10) << speedup << "x" << std::endl;
        std::cout << "   Выигрыш:             " << std::setw(10) 
                  << ((speedup - 1) * 100) << "%" << std::endl;
        
        // записываем в файл
        resultsFile << SIZE << "," << globalTime << "," << sharedTime << "," 
                   << speedup << std::endl;
        
        // освобождаем память
        cudaFree(d_input); // освобождаем
    }
    
    // закрываем файл
    resultsFile.close(); // закрываем
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" << std::endl;
    std::cout << "Результаты сохранены в файл: performance_results.csv" << std::endl;
    std::cout << std::string(70, '=') << std::endl << std::endl;
    
    std::cout << "Инструкция для построения графиков:" << std::endl;
    std::cout << "1. Откройте файл performance_results.csv в Excel/Google Sheets" << std::endl;
    std::cout << "2. Выделите данные и создайте график" << std::endl;
    std::cout << "3. Тип графика: линейный график" << std::endl;
    std::cout << "4. Ось X: Размер массива" << std::endl;
    std::cout << "5. Ось Y: Время выполнения (мс)" << std::endl;
    std::cout << "6. Две линии: для глобальной и разделяемой памяти" << std::endl;
    
    return 0; // завершение
}
