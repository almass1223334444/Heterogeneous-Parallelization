// задание 3: сортировка массива на GPU
// используем пузырьковую сортировку для подмассивов и слияние

#include <iostream> // для вывода
#include <cuda_runtime.h> // для CUDA
#include <vector> // для массивов
#include <fstream> // для файлов
#include <algorithm> // для std::sort
#include <chrono> // для времени

// kernel для пузырьковой сортировки небольших подмассивов
// каждый блок сортирует свой подмассив
__global__ void bubbleSortKernel(float* data, int size, int chunkSize) {
    // вычисляем начало подмассива для этого блока
    int start = blockIdx.x * chunkSize; // начальный индекс
    int end = min(start + chunkSize, size); // конечный индекс (не выходим за границы)
    
    // только первый поток блока выполняет сортировку
    if (threadIdx.x == 0) {
        // пузырьковая сортировка - проходим по массиву много раз
        for (int i = start; i < end - 1; ++i) { // внешний цикл
            for (int j = start; j < end - 1 - (i - start); ++j) { // внутренний цикл
                // если элементы стоят не в порядке, меняем их местами
                if (data[j] > data[j + 1]) {
                    // обмен значений
                    float temp = data[j]; // сохраняем в временную переменную
                    data[j] = data[j + 1]; // перезаписываем первый элемент
                    data[j + 1] = temp; // записываем сохраненное значение
                }
            }
        }
    }
}

// kernel для слияния отсортированных подмассивов с разделяемой памятью
__global__ void mergeKernel(float* input, float* output, int size, int chunkSize) {
    // выделяем разделяемую память для двух подмассивов
    extern __shared__ float shared[]; // динамически выделяемая разделяемая память
    
    // вычисляем индексы для слияния
    int tid = threadIdx.x; // локальный индекс потока
    int mergeId = blockIdx.x; // номер операции слияния
    
    // начала двух сливаемых подмассивов
    int start1 = mergeId * 2 * chunkSize; // начало первого подмассива
    int end1 = min(start1 + chunkSize, size); // конец первого
    int start2 = end1; // начало второго (сразу после первого)
    int end2 = min(start2 + chunkSize, size); // конец второго
    
    int size1 = end1 - start1; // размер первого подмассива
    int size2 = end2 - start2; // размер второго подмассива
    
    // загружаем данные в разделяемую память
    // каждый поток загружает один элемент из первого подмассива
    if (tid < size1) {
        shared[tid] = input[start1 + tid]; // копируем в первую часть shared
    }
    // и один элемент из второго подмассива
    if (tid < size2) {
        shared[size1 + tid] = input[start2 + tid]; // копируем во вторую часть shared
    }
    
    // ждем завершения загрузки всеми потоками
    __syncthreads(); // синхронизация
    
    // слияние двух отсортированных подмассивов
    // каждый поток вычисляет позицию для одного элемента результата
    if (tid < size1 + size2) {
        int i = 0; // индекс в первом подмассиве
        int j = 0; // индекс во втором подмассиве
        int k = tid; // позиция в результате
        
        // простое слияние - находим k-й по порядку элемент
        int count = 0; // счетчик элементов
        float value; // значение для записи
        
        // перебираем элементы пока не найдем нужный
        while (count <= k && (i < size1 || j < size2)) {
            // берем меньший элемент из двух подмассивов
            if (i < size1 && (j >= size2 || shared[i] <= shared[size1 + j])) {
                value = shared[i]; // берем из первого
                i++; // сдвигаем указатель
            } else {
                value = shared[size1 + j]; // берем из второго
                j++; // сдвигаем указатель
            }
            
            // если это k-й элемент, записываем его
            if (count == k) {
                output[start1 + tid] = value; // записываем в результат
                break; // выходим из цикла
            }
            count++; // увеличиваем счетчик
        }
    }
}

// функция загрузки данных
void loadData(std::vector<float>& data, int size) {
    std::ifstream inFile("random_data.bin", std::ios::binary); // открываем файл
    if (!inFile) { // если не открылся
        std::cerr << "Ошибка открытия файла!" << std::endl;
        exit(1); // выход
    }
    data.resize(size); // устанавливаем размер
    inFile.read(reinterpret_cast<char*>(data.data()), size * sizeof(float)); // читаем
    inFile.close(); // закрываем
}

// функция проверки сортировки
bool isSorted(const std::vector<float>& arr) {
    // проходим по массиву и проверяем порядок
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) { // если нарушен порядок
            return false; // массив не отсортирован
        }
    }
    return true; // массив отсортирован
}

int main() {
    // размеры для тестирования
    int sizes[] = {10000, 100000}; // берем поменьше, т.к. сортировка медленная
    
    // проходим по размерам
    for (int s = 0; s < 2; ++s) {
        int SIZE = sizes[s]; // текущий размер
        
        std::cout << "\n=== Тест сортировки для размера: " << SIZE << " ===" << std::endl;
        
        // загружаем данные
        std::vector<float> h_data; // данные на CPU
        loadData(h_data, SIZE); // загружаем
        
        // создаем копию для CPU сортировки
        std::vector<float> h_data_cpu = h_data; // копируем
        
        // сортируем на CPU для проверки
        auto cpu_start = std::chrono::high_resolution_clock::now(); // начало отсчета
        std::sort(h_data_cpu.begin(), h_data_cpu.end()); // стандартная сортировка
        auto cpu_end = std::chrono::high_resolution_clock::now(); // конец отсчета
        
        // вычисляем время CPU
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        std::cout << "Время CPU сортировки: " << cpu_duration.count() << " мс" << std::endl;
        
        // выделяем память на GPU
        float* d_data; // данные на GPU
        float* d_temp; // временный массив для слияния
        
        cudaMalloc(&d_data, SIZE * sizeof(float)); // выделяем основной массив
        cudaMalloc(&d_temp, SIZE * sizeof(float)); // выделяем временный
        
        // копируем данные на GPU
        cudaMemcpy(d_data, h_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
        
        // параметры для сортировки
        int chunkSize = 1024; // размер подмассива для сортировки (не больше, иначе долго)
        int numChunks = (SIZE + chunkSize - 1) / chunkSize; // количество подмассивов
        
        // создаем события
        cudaEvent_t start, stop; // события
        cudaEventCreate(&start); // создаем
        cudaEventCreate(&stop); // создаем
        
        cudaEventRecord(start); // начало
        
        // первый этап - сортируем каждый подмассив
        bubbleSortKernel<<<numChunks, 1>>>(d_data, SIZE, chunkSize);
        
        // второй этап - сливаем отсортированные подмассивы
        int currentChunkSize = chunkSize; // текущий размер подмассивов
        float* currentInput = d_data; // текущий входной массив
        float* currentOutput = d_temp; // текущий выходной массив
        
        // сливаем пока не останется один массив
        while (currentChunkSize < SIZE) {
            // количество операций слияния
            int numMerges = (SIZE + 2 * currentChunkSize - 1) / (2 * currentChunkSize);
            
            // размер разделяемой памяти - два подмассива
            int sharedMemSize = 2 * currentChunkSize * sizeof(float);
            
            // запускаем слияние
            mergeKernel<<<numMerges, currentChunkSize * 2, sharedMemSize>>>(
                currentInput, currentOutput, SIZE, currentChunkSize
            );
            
            // меняем местами входной и выходной массивы
            float* temp = currentInput; // сохраняем указатель
            currentInput = currentOutput; // выход становится входом
            currentOutput = temp; // вход становится выходом
            
            // удваиваем размер подмассивов
            currentChunkSize *= 2; // в два раза больше
        }
        
        cudaEventRecord(stop); // конец
        cudaEventSynchronize(stop); // ждем
        
        // получаем время
        float milliseconds = 0; // время
        cudaEventElapsedTime(&milliseconds, start, stop); // вычисляем
        
        std::cout << "Время GPU сортировки: " << milliseconds << " мс" << std::endl;
        
        // копируем результат обратно
        cudaMemcpy(h_data.data(), currentInput, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        // проверяем правильность
        if (isSorted(h_data)) {
            std::cout << "✓ Массив отсортирован правильно!" << std::endl;
        } else {
            std::cout << "✗ Ошибка сортировки!" << std::endl;
        }
        
        // сравниваем первые 10 элементов с CPU
        std::cout << "Первые 10 элементов GPU: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << h_data[i] << " "; // вывод
        }
        std::cout << std::endl;
        
        // освобождаем память
        cudaFree(d_data); // освобождаем основной массив
        cudaFree(d_temp); // освобождаем временный
        
        // удаляем события
        cudaEventDestroy(start); // удаляем
        cudaEventDestroy(stop); // удаляем
    }
    
    return 0; // завершение
}
