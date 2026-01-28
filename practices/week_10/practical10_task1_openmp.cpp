#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// функция вычисления суммы массива параллельно
double parallelSum(const std::vector<double>& data, int numThreads) {
    double sum = 0.0;
    
    // устанавливаем количество потоков
    omp_set_num_threads(numThreads);
    
    // параллельная область с редукцией для суммирования
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    
    return sum;
}

// функция вычисления среднего значения
double parallelMean(const std::vector<double>& data, int numThreads) {
    double sum = parallelSum(data, numThreads);
    // деление на размер массива (последовательная часть)
    return sum / data.size();
}

// функция вычисления дисперсии параллельно
double parallelVariance(const std::vector<double>& data, double mean, int numThreads) {
    double variance = 0.0;
    
    // устанавливаем количество потоков
    omp_set_num_threads(numThreads);
    
    // параллельное вычисление суммы квадратов отклонений
    #pragma omp parallel for reduction(+:variance)
    for (size_t i = 0; i < data.size(); i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    
    // деление на размер массива (последовательная часть)
    return variance / data.size();
}

// последовательная версия для сравнения
void sequentialCompute(const std::vector<double>& data, double& sum, double& mean, double& variance) {
    // вычисление суммы
    sum = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    
    // вычисление среднего
    mean = sum / data.size();
    
    // вычисление дисперсии
    variance = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double diff = data[i] - mean;
        variance += diff * diff;
    }
    variance /= data.size();
}

int main() {
    // размер массива для тестирования
    const size_t arraySize = 100000000;  // 100 миллионов элементов
    
    std::cout << "Создание массива из " << arraySize << " элементов...\n";
    
    // создаём и инициализируем массив
    std::vector<double> data(arraySize);
    for (size_t i = 0; i < arraySize; i++) {
        data[i] = static_cast<double>(i % 1000) / 10.0;
    }
    
    std::cout << "Массив создан\n\n";
    
    // --- последовательная версия ---
    std::cout << "=== ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ===\n";
    double seqSum, seqMean, seqVariance;
    
    // засекаем время начала
    double startTime = omp_get_wtime();
    
    sequentialCompute(data, seqSum, seqMean, seqVariance);
    
    // засекаем время окончания
    double endTime = omp_get_wtime();
    double seqTime = endTime - startTime;
    
    std::cout << "Время выполнения: " << seqTime << " сек\n";
    std::cout << "Сумма: " << seqSum << "\n";
    std::cout << "Среднее: " << seqMean << "\n";
    std::cout << "Дисперсия: " << seqVariance << "\n\n";
    
    // --- тестирование с разным количеством потоков ---
    std::cout << "=== ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP) ===\n";
    std::cout << "Потоки | Время (сек) | Ускорение | Эффективность | Доля парал.\n";
    std::cout << "-------|-------------|-----------|---------------|-------------\n";
    
    // массив для тестирования разного количества потоков
    int threadCounts[] = {1, 2, 4, 8, 16};
    
    for (int numThreads : threadCounts) {
        // засекаем время для параллельной версии
        startTime = omp_get_wtime();
        
        // вычисляем сумму
        double parSum = parallelSum(data, numThreads);
        
        // вычисляем среднее
        double parMean = parallelMean(data, numThreads);
        
        // вычисляем дисперсию
        double parVariance = parallelVariance(data, parMean, numThreads);
        
        endTime = omp_get_wtime();
        double parTime = endTime - startTime;
        
        // вычисляем метрики производительности
        double speedup = seqTime / parTime;  // ускорение
        double efficiency = speedup / numThreads;  // эффективность
        
        // вычисляем долю параллельной части по закону Амдала
        // S = 1 / ((1-p) + p/n), где S - ускорение, p - доля параллельной части, n - число потоков
        // решаем относительно p: p = (n * (S - 1)) / (S * (n - 1))
        double parallelFraction = 0.0;
        if (numThreads > 1 && speedup > 1.0) {
            parallelFraction = (numThreads * (speedup - 1.0)) / (speedup * (numThreads - 1.0));
        }
        
        std::cout << numThreads << "      | "
                  << parTime << " | "
                  << speedup << "x   | "
                  << (efficiency * 100) << "%      | "
                  << (parallelFraction * 100) << "%\n";
    }
    
    std::cout << "\n=== АНАЛИЗ ПО ЗАКОНУ АМДАЛА ===\n";
    std::cout << "Закон Амдала: S = 1 / ((1-p) + p/n)\n";
    std::cout << "где S - ускорение, p - доля параллельной части, n - число потоков\n\n";
    
    // теоретическое максимальное ускорение при разных долях параллельной части
    std::cout << "Доля парал. части | Макс. ускорение (теор.)\n";
    std::cout << "------------------|------------------------\n";
    for (double p : {0.5, 0.75, 0.9, 0.95, 0.99}) {
        double maxSpeedup = 1.0 / (1.0 - p);  // при бесконечном числе потоков
        std::cout << (p * 100) << "%           | " << maxSpeedup << "x\n";
    }
    
    std::cout << "\n=== ПРОФИЛИРОВАНИЕ ===\n";
    std::cout << "Последовательная часть программы:\n";
    std::cout << "  - Инициализация массива\n";
    std::cout << "  - Деление на размер массива (в mean и variance)\n";
    std::cout << "  - Вывод результатов\n\n";
    
    std::cout << "Параллельная часть программы:\n";
    std::cout << "  - Циклы вычисления суммы\n";
    std::cout << "  - Циклы вычисления дисперсии\n\n";
    
    std::cout << "ВЫВОДЫ:\n";
    std::cout << "1. При увеличении потоков ускорение растёт, но не линейно\n";
    std::cout << "2. Эффективность использования потоков падает с их ростом\n";
    std::cout << "3. Последовательные части (инициализация, деление) ограничивают максимальное ускорение\n";
    std::cout << "4. Накладные расходы на создание потоков и синхронизацию снижают производительность\n";
    
    return 0;
}
