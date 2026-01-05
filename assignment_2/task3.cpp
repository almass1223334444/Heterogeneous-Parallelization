// ============================================================================
// Task 3: Selection Sort (Sequential vs OpenMP Parallel)
// Требование: каждая строка закомментирована
// ============================================================================

#include <iostream>      // ввод / вывод (cout)
#include <vector>        // std::vector
#include <random>        // генерация случайных чисел
#include <chrono>        // измерение времени

#ifdef _OPENMP           // если OpenMP доступен
#include <omp.h>         // функции OpenMP
#endif                   // конец условия

using namespace std;     // используем стандартное пространство имён

// ============================================================================
// Функция генерации массива случайных чисел
// ============================================================================

vector<int> generateArray(int n) {                          // создаёт массив размера n
    vector<int> a(n);                                       // выделяем память под массив

    random_device rd;                                       // источник энтропии
    mt19937 gen(rd());                                      // генератор mt19937
    uniform_int_distribution<int> dist(1, 10000);           // диапазон значений [1..10000]

    for (int i = 0; i < n; ++i) {                           // цикл по массиву
        a[i] = dist(gen);                                   // заполняем случайным числом
    }                                                       // конец цикла

    return a;                                               // возвращаем массив
}                                                           // конец функции

// ============================================================================
// Последовательная сортировка выбором
// ============================================================================

void selectionSortSequential(vector<int>& a) {              // функция последовательной сортировки
    int n = a.size();                                       // размер массива

    for (int i = 0; i < n - 1; ++i) {                        // позиция текущего минимума
        int minIndex = i;                                   // предполагаем минимумом текущий i

        for (int j = i + 1; j < n; ++j) {                   // ищем минимум справа
            if (a[j] < a[minIndex]) {                       // если нашли меньший элемент
                minIndex = j;                               // обновляем индекс минимума
            }                                               // конец if
        }                                                   // конец внутреннего цикла

        swap(a[i], a[minIndex]);                            // меняем местами
    }                                                       // конец внешнего цикла
}                                                           // конец функции

// ============================================================================
// Параллельная сортировка выбором (параллельный поиск минимума)
// ============================================================================

void selectionSortParallel(vector<int>& a) {                // функция параллельной сортировки
    int n = a.size();                                       // размер массива

    for (int i = 0; i < n - 1; ++i) {                        // текущая позиция сортировки
        int globalMinVal = a[i];                             // глобальный минимум
        int globalMinIdx = i;                                // индекс глобального минимума

#ifdef _OPENMP                                              // если OpenMP доступен
        #pragma omp parallel                                 // создаём параллельную область
        {
            int localMinVal = globalMinVal;                  // локальный минимум потока
            int localMinIdx = globalMinIdx;                  // индекс локального минимума

            #pragma omp for nowait schedule(static)          // делим цикл между потоками
            for (int j = i + 1; j < n; ++j) {                // каждый поток проверяет свой диапазон
                if (a[j] < localMinVal) {                    // если нашли меньший элемент
                    localMinVal = a[j];                      // обновляем локальный минимум
                    localMinIdx = j;                         // обновляем индекс
                }                                           // конец if
            }                                               // конец omp for

            #pragma omp critical                              // критическая секция
            {
                if (localMinVal < globalMinVal) {            // сравниваем с глобальным минимумом
                    globalMinVal = localMinVal;              // обновляем глобальное значение
                    globalMinIdx = localMinIdx;              // обновляем глобальный индекс
                }                                           // конец if
            }                                               // конец critical
        }                                                   // конец parallel
#else                                                       // если OpenMP недоступен
        for (int j = i + 1; j < n; ++j) {                    // обычный последовательный цикл
            if (a[j] < globalMinVal) {                       // поиск минимума
                globalMinVal = a[j];                         // обновляем значение
                globalMinIdx = j;                            // обновляем индекс
            }                                               // конец if
        }                                                   // конец цикла
#endif                                                      // конец условия OpenMP

        swap(a[i], a[globalMinIdx]);                         // ставим минимум на позицию i
    }                                                       // конец внешнего цикла
}                                                           // конец функции

// ============================================================================
// Функция измерения времени выполнения сортировки
// ============================================================================

double measureTime(vector<int> a, void (*sortFunc)(vector<int>&)) { // принимает копию массива
    auto start = chrono::high_resolution_clock::now();      // старт таймера
    sortFunc(a);                                             // вызываем сортировку
    auto end = chrono::high_resolution_clock::now();        // конец таймера
    chrono::duration<double, milli> dur = end - start;      // считаем время
    return dur.count();                                      // возвращаем время в мс
}                                                           // конец функции

// ============================================================================
// main
// ============================================================================

int main() {                                                 // точка входа программы
    cout << "TASK 3 — Selection Sort (Sequential vs OpenMP)\n"; // заголовок

#ifdef _OPENMP                                              // если OpenMP включён
    cout << "OpenMP enabled, threads = " << omp_get_max_threads() << "\n"; // вывод потоков
#else                                                       // если OpenMP выключен
    cout << "OpenMP not enabled\n";                          // предупреждение
#endif                                                      // конец условия

    // --- Тест 1: массив 1000 элементов ---
    int n1 = 1000;                                          // первый размер массива
    vector<int> a1 = generateArray(n1);                     // генерация массива

    double t_seq_1 = measureTime(a1, selectionSortSequential); // время последовательной версии
    double t_par_1 = measureTime(a1, selectionSortParallel);   // время параллельной версии

    cout << "\nArray size: " << n1 << "\n";                 // вывод размера
    cout << "Sequential time: " << t_seq_1 << " ms\n";      // вывод времени
    cout << "Parallel time:   " << t_par_1 << " ms\n";      // вывод времени

    // --- Тест 2: массив 10000 элементов ---
    int n2 = 10000;                                         // второй размер массива
    vector<int> a2 = generateArray(n2);                     // генерация массива

    double t_seq_2 = measureTime(a2, selectionSortSequential); // время последовательной версии
    double t_par_2 = measureTime(a2, selectionSortParallel);   // время параллельной версии

    cout << "\nArray size: " << n2 << "\n";                 // вывод размера
    cout << "Sequential time: " << t_seq_2 << " ms\n";      // вывод времени
    cout << "Parallel time:   " << t_par_2 << " ms\n";      // вывод времени

    cout << "\nConclusion:\n";                              // вывод заключения
    cout << "Selection sort has limited parallelism.\n";   // объяснение
    cout << "Parallel version may be slower due to synchronization overhead.\n"; // вывод

    return 0;                                               // завершение программы
}                                                           // конец main
