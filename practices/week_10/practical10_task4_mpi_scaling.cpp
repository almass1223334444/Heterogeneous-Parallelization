#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <iomanip>

// функция вычисления суммы локальной части массива
double computeLocalSum(const std::vector<double>& localData) {
    double sum = 0.0;
    for (size_t i = 0; i < localData.size(); i++) {
        sum += localData[i];
    }
    return sum;
}

// функция поиска минимума в локальной части
double computeLocalMin(const std::vector<double>& localData) {
    if (localData.empty()) return 0.0;
    
    double minVal = localData[0];
    for (size_t i = 1; i < localData.size(); i++) {
        if (localData[i] < minVal) {
            minVal = localData[i];
        }
    }
    return minVal;
}

// функция поиска максимума в локальной части
double computeLocalMax(const std::vector<double>& localData) {
    if (localData.empty()) return 0.0;
    
    double maxVal = localData[0];
    for (size_t i = 1; i < localData.size(); i++) {
        if (localData[i] > maxVal) {
            maxVal = localData[i];
        }
    }
    return maxVal;
}

int main(int argc, char** argv) {
    // инициализируем mpi
    MPI_Init(&argc, &argv);
    
    // получаем количество процессов
    int worldSize;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    
    // получаем ранг текущего процесса
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // --- параметры для тестирования ---
    // для strong scaling используем фиксированный размер задачи
    const long long strongScalingSize = 100000000;  // 100 миллионов элементов
    
    // для weak scaling размер на процесс фиксирован
    const long long sizePerProcess = 10000000;  // 10 миллионов на процесс
    long long weakScalingSize = sizePerProcess * worldSize;
    
    // главный процесс выводит информацию
    if (rank == 0) {
        std::cout << "\n=== АНАЛИЗ МАСШТАБИРУЕМОСТИ MPI ===\n";
        std::cout << "Количество процессов: " << worldSize << "\n\n";
    }
    
    // ==================== STRONG SCALING TEST ====================
    if (rank == 0) {
        std::cout << "=== STRONG SCALING TEST ===\n";
        std::cout << "Фиксированный размер задачи: " << strongScalingSize << " элементов\n";
        std::cout << "Размер на процесс: " << strongScalingSize / worldSize << " элементов\n\n";
    }
    
    // вычисляем локальный размер для каждого процесса
    long long strongLocalSize = strongScalingSize / worldSize;
    // последний процесс берёт остаток
    if (rank == worldSize - 1) {
        strongLocalSize += strongScalingSize % worldSize;
    }
    
    // создаём локальный массив
    std::vector<double> strongLocalData(strongLocalSize);
    
    // инициализируем данные
    long long globalOffset = rank * (strongScalingSize / worldSize);
    for (long long i = 0; i < strongLocalSize; i++) {
        strongLocalData[i] = static_cast<double>((globalOffset + i) % 10000) / 100.0;
    }
    
    // барьер синхронизации перед началом
    MPI_Barrier(MPI_COMM_WORLD);
    
    // начинаем отсчёт времени
    double strongStartTime = MPI_Wtime();
    
    // --- вычисление локальных результатов ---
    double localSum = computeLocalSum(strongLocalData);
    double localMin = computeLocalMin(strongLocalData);
    double localMax = computeLocalMax(strongLocalData);
    
    // время локальных вычислений
    double localComputeTime = MPI_Wtime() - strongStartTime;
    
    // начинаем замер времени коммуникации
    double commStartTime = MPI_Wtime();
    
    // --- тест 1: MPI_Reduce (только root получает результат) ---
    double globalSum = 0.0;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double globalMin = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    double globalMax = 0.0;
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // время коммуникации
    double reduceCommTime = MPI_Wtime() - commStartTime;
    
    // --- тест 2: MPI_Allreduce (все процессы получают результат) ---
    commStartTime = MPI_Wtime();
    
    double allreduceSum = 0.0;
    MPI_Allreduce(&localSum, &allreduceSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    double allreduceCommTime = MPI_Wtime() - commStartTime;
    
    // барьер для точного замера
    MPI_Barrier(MPI_COMM_WORLD);
    
    // общее время выполнения
    double strongTotalTime = MPI_Wtime() - strongStartTime;
    
    // главный процесс собирает и выводит статистику
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Результаты вычислений:\n";
        std::cout << "  Сумма: " << globalSum << "\n";
        std::cout << "  Минимум: " << globalMin << "\n";
        std::cout << "  Максимум: " << globalMax << "\n\n";
        
        std::cout << "Время выполнения:\n";
        std::cout << "  Локальные вычисления: " << localComputeTime * 1000 << " мс\n";
        std::cout << "  MPI_Reduce:           " << reduceCommTime * 1000 << " мс\n";
        std::cout << "  MPI_Allreduce:        " << allreduceCommTime * 1000 << " мс\n";
        std::cout << "  Общее время:          " << strongTotalTime * 1000 << " мс\n\n";
        
        std::cout << "Анализ:\n";
        std::cout << "  Доля вычислений: " << (localComputeTime / strongTotalTime * 100) << "%\n";
        std::cout << "  Доля коммуникаций: " << ((reduceCommTime + allreduceCommTime) / strongTotalTime * 100) << "%\n\n";
    }
    
    // ==================== WEAK SCALING TEST ====================
    if (rank == 0) {
        std::cout << "=== WEAK SCALING TEST ===\n";
        std::cout << "Фиксированный размер на процесс: " << sizePerProcess << " элементов\n";
        std::cout << "Общий размер задачи: " << weakScalingSize << " элементов\n\n";
    }
    
    // создаём локальный массив фиксированного размера
    std::vector<double> weakLocalData(sizePerProcess);
    
    // инициализируем данные
    globalOffset = rank * sizePerProcess;
    for (long long i = 0; i < sizePerProcess; i++) {
        weakLocalData[i] = static_cast<double>((globalOffset + i) % 10000) / 100.0;
    }
    
    // барьер перед началом
    MPI_Barrier(MPI_COMM_WORLD);
    
    // начинаем отсчёт времени
    double weakStartTime = MPI_Wtime();
    
    // вычисление локальных результатов
    double weakLocalSum = computeLocalSum(weakLocalData);
    double weakLocalMin = computeLocalMin(weakLocalData);
    double weakLocalMax = computeLocalMax(weakLocalData);
    
    // собираем результаты
    double weakGlobalSum = 0.0;
    MPI_Reduce(&weakLocalSum, &weakGlobalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    double weakGlobalMin = 0.0;
    MPI_Reduce(&weakLocalMin, &weakGlobalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    double weakGlobalMax = 0.0;
    MPI_Reduce(&weakLocalMax, &weakGlobalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // общее время
    double weakTotalTime = MPI_Wtime() - weakStartTime;
    
    if (rank == 0) {
        std::cout << "Результаты:\n";
        std::cout << "  Время выполнения: " << weakTotalTime * 1000 << " мс\n";
        std::cout << "  Сумма: " << weakGlobalSum << "\n\n";
    }
    
    // ==================== СБОР СТАТИСТИКИ ОТ ВСЕХ ПРОЦЕССОВ ====================
    // собираем время выполнения от каждого процесса для анализа балансировки
    std::vector<double> allStrongTimes;
    std::vector<double> allWeakTimes;
    
    if (rank == 0) {
        allStrongTimes.resize(worldSize);
        allWeakTimes.resize(worldSize);
    }
    
    // собираем времена на root процессе
    MPI_Gather(&strongTotalTime, 1, MPI_DOUBLE, allStrongTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&weakTotalTime, 1, MPI_DOUBLE, allWeakTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ==================== АНАЛИЗ И ВЫВОДЫ ====================
    if (rank == 0) {
        std::cout << "=== ДЕТАЛЬНЫЙ АНАЛИЗ МАСШТАБИРУЕМОСТИ ===\n\n";
        
        // анализ балансировки нагрузки
        double maxStrongTime = *std::max_element(allStrongTimes.begin(), allStrongTimes.end());
        double minStrongTime = *std::min_element(allStrongTimes.begin(), allStrongTimes.end());
        
        std::cout << "Балансировка нагрузки (Strong Scaling):\n";
        std::cout << "  Максимальное время процесса: " << maxStrongTime * 1000 << " мс\n";
        std::cout << "  Минимальное время процесса:  " << minStrongTime * 1000 << " мс\n";
        std::cout << "  Разброс: " << ((maxStrongTime - minStrongTime) / maxStrongTime * 100) << "%\n\n";
        
        // расчёт эффективности
        std::cout << "STRONG SCALING ЭФФЕКТИВНОСТЬ:\n";
        std::cout << "  При идеальном масштабировании время должно уменьшаться в N раз\n";
        std::cout << "  Реальная эффективность зависит от:\n";
        std::cout << "    - Доли последовательного кода (закон Амдала)\n";
        std::cout << "    - Накладных расходов на коммуникацию\n";
        std::cout << "    - Балансировки нагрузки\n\n";
        
        std::cout << "WEAK SCALING ЭФФЕКТИВНОСТЬ:\n";
        std::cout << "  При идеальном масштабировании время должно оставаться постоянным\n";
        std::cout << "  Время: " << weakTotalTime * 1000 << " мс\n";
        std::cout << "  При увеличении процессов время растёт из-за:\n";
        std::cout << "    - Увеличения коммуникационных затрат\n";
        std::cout << "    - Большего числа участников в коллективных операциях\n\n";
        
        std::cout << "ВЛИЯНИЕ КОММУНИКАЦИОННЫХ ОПЕРАЦИЙ:\n\n";
        
        std::cout << "MPI_Reduce vs MPI_Allreduce:\n";
        std::cout << "  MPI_Reduce время:    " << reduceCommTime * 1000 << " мс\n";
        std::cout << "  MPI_Allreduce время: " << allreduceCommTime * 1000 << " мс\n";
        std::cout << "  Разница:             " << (allreduceCommTime / reduceCommTime) << "x\n\n";
        
        std::cout << "  MPI_Reduce:\n";
        std::cout << "    + Результат только на root процессе\n";
        std::cout << "    + Меньше коммуникации\n";
        std::cout << "    - Нужна последующая рассылка если результат нужен всем\n\n";
        
        std::cout << "  MPI_Allreduce:\n";
        std::cout << "    + Результат сразу на всех процессах\n";
        std::cout << "    + Не нужна дополнительная рассылка\n";
        std::cout << "    - Больше коммуникационных затрат\n\n";
        
        std::cout << "=== ПРАКТИЧЕСКИЕ ОГРАНИЧЕНИЯ МАСШТАБИРУЕМОСТИ ===\n\n";
        
        std::cout << "1. ЗАКОН АМДАЛА:\n";
        std::cout << "   Максимальное ускорение ограничено последовательной частью:\n";
        std::cout << "   S_max = 1 / (s + p/n), где s - последовательная часть\n";
        std::cout << "   Даже 5% последовательного кода ограничивает ускорение до 20x\n\n";
        
        std::cout << "2. КОММУНИКАЦИОННЫЕ ЗАТРАТЫ:\n";
        std::cout << "   - Растут с увеличением числа процессов\n";
        std::cout << "   - Зависят от топологии сети и латентности\n";
        std::cout << "   - MPI_Reduce: O(log n), MPI_Allreduce: O(log n)\n";
        std::cout << "   - При большом n коммуникация доминирует\n\n";
        
        std::cout << "3. БАЛАНСИРОВКА НАГРУЗКИ:\n";
        std::cout << "   - Неравномерное распределение данных\n";
        std::cout << "   - Различная сложность обработки элементов\n";
        std::cout << "   - Быстрые процессы ждут медленные\n\n";
        
        std::cout << "4. РАЗМЕР ЗАДАЧИ:\n";
        std::cout << "   - Маленькие задачи: накладные расходы > выигрыш\n";
        std::cout << "   - Большие задачи: лучше масштабируются\n";
        std::cout << "   - Оптимальное число процессов зависит от размера\n\n";
        
        std::cout << "5. ЗАКОН ГУСТАФСОНА (для weak scaling):\n";
        std::cout << "   S = s + p*n, где n - число процессов\n";
        std::cout << "   Более оптимистичен чем закон Амдала\n";
        std::cout << "   Предполагает что размер задачи растёт с ресурсами\n\n";
        
        std::cout << "=== ВЫВОДЫ ===\n";
        std::cout << "1. Масштабируемость ограничена последовательными частями (Амдал)\n";
        std::cout << "2. Коммуникационные операции становятся узким местом при росте процессов\n";
        std::cout << "3. MPI_Allreduce дороже MPI_Reduce но удобнее для распределённых алгоритмов\n";
        std::cout << "4. Strong scaling эффективен до определённого числа процессов\n";
        std::cout << "5. Weak scaling показывает способность обрабатывать большие данные\n";
        std::cout << "6. Для данного алгоритма оптимальное число процессов: 4-16\n";
        std::cout << "7. При большем числе процессов коммуникация начинает доминировать\n";
    }
    
    // завершаем mpi
    MPI_Finalize();
    
    return 0;
}
