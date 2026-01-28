// задание 2a: редукция суммы только через глобальную память
// программа суммирует все элементы массива используя GPU

#include <iostream> // для вывода
#include <cuda_runtime.h> // для работы с CUDA
#include <vector> // для массивов
#include <fstream> // для чтения файла
#include <chrono> // для измерения времени

// kernel функция - выполняется на GPU
// каждый поток суммирует свою часть массива
__global__ void reduceGlobalMemory(float* input, float* output, int size) {
    // вычисляем глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // номер потока в общей сетке
    
    // stride - шаг между элементами, которые обрабатывает один поток
    int stride = blockDim.x * gridDim.x; // общее количество потоков
    
    // каждый поток суммирует элементы с шагом stride
    float sum = 0.0f; // локальная сумма для этого потока
    for (int i = tid; i < size; i += stride) {
        sum += input[i]; // добавляем элемент к сумме
    }
    
    // записываем результат в глобальную память
    output[tid] = sum; // каждый поток записывает свою частичную сумму
}

// функция для загрузки данных из файла
void loadData(std::vector<float>& data, int size) {
    std::ifstream inFile("random_data.bin", std::ios::binary); // открываем файл
    if (!inFile) { // проверяем успешность открытия
        std::cerr << "Ошибка открытия файла!" << std::endl;
        exit(1); // завершаем программу с ошибкой
    }
    data.resize(size); // устанавливаем размер вектора
    inFile.read(reinterpret_cast<char*>(data.data()), size * sizeof(float)); // читаем данные
    inFile.close(); // закрываем файл
}

int main() {
    // размеры массивов для тестирования
    int sizes[] = {10000, 100000, 1000000}; // три разных размера
    
    // проходим по всем размерам
    for (int s = 0; s < 3; ++s) {
        int SIZE = sizes[s]; // текущий размер
        
        std::cout << "\n=== Тест для размера: " << SIZE << " ===" << std::endl;
        
        // загружаем данные с диска
        std::vector<float> h_data; // массив на CPU (host)
        loadData(h_data, SIZE); // загружаем данные
        
        // вычисляем эталонную сумму на CPU для проверки
        float cpu_sum = 0.0f; // переменная для суммы
        for (int i = 0; i < SIZE; ++i) {
            cpu_sum += h_data[i]; // суммируем на CPU
        }
        std::cout << "Эталонная сумма (CPU): " << cpu_sum << std::endl;
        
        // выделяем память на GPU
        float* d_input; // указатель на входные данные на GPU
        float* d_output; // указатель на выходные данные на GPU
        
        cudaMalloc(&d_input, SIZE * sizeof(float)); // выделяем память под входной массив
        cudaMalloc(&d_output, 1024 * sizeof(float)); // память под частичные суммы (максимум 1024 блока)
        
        // копируем данные с CPU на GPU
        cudaMemcpy(d_input, h_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        // настраиваем параметры запуска kernel
        int threadsPerBlock = 256; // количество потоков в блоке
        int blocks = 256; // количество блоков
        
        // создаем события для измерения времени
        cudaEvent_t start, stop; // события начала и конца
        cudaEventCreate(&start); // создаем событие начала
        cudaEventCreate(&stop); // создаем событие конца
        
        // засекаем время начала
        cudaEventRecord(start); // записываем время старта
        
        // запускаем kernel на GPU
        reduceGlobalMemory<<<blocks, threadsPerBlock>>>(d_input, d_output, SIZE);
        
        // засекаем время окончания
        cudaEventRecord(stop); // записываем время окончания
        cudaEventSynchronize(stop); // ждем завершения всех операций
        
        // вычисляем затраченное время
        float milliseconds = 0; // переменная для времени в миллисекундах
        cudaEventElapsedTime(&milliseconds, start, stop); // получаем разницу времени
        
        // копируем частичные результаты обратно на CPU
        std::vector<float> h_output(blocks * threadsPerBlock); // массив для частичных сумм
        cudaMemcpy(h_output.data(), d_output, blocks * threadsPerBlock * sizeof(float), cudaMemcpyDeviceToHost);
        
        // суммируем частичные результаты на CPU
        float gpu_sum = 0.0f; // итоговая сумма
        for (int i = 0; i < blocks * threadsPerBlock; ++i) {
            gpu_sum += h_output[i]; // складываем все частичные суммы
        }
        
        // выводим результаты
        std::cout << "Сумма на GPU (глобальная память): " << gpu_sum << std::endl;
        std::cout << "Время выполнения: " << milliseconds << " мс" << std::endl;
        std::cout << "Разница с CPU: " << std::abs(cpu_sum - gpu_sum) << std::endl;
        
        // освобождаем память на GPU
        cudaFree(d_input); // освобождаем входной массив
        cudaFree(d_output); // освобождаем выходной массив
        
        // уничтожаем события
        cudaEventDestroy(start); // удаляем событие начала
        cudaEventDestroy(stop); // удаляем событие конца
    }
    
    return 0; // программа завершена
}
