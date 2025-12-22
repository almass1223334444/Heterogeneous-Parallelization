// lab3.cpp
// Часть 3: Динамическая память, указатели и параллельный подсчёт среднего
//
// 1) Создаём динамический массив через new[]
// 2) Заполняем случайными числами
// 3) Считаем среднее значение:
//    - последовательно
//    - параллельно (OpenMP + reduction)
// 4) Освобождаем память (delete[])
//
// Компиляция (PowerShell, MinGW):
// g++ -fopenmp -O2 -std=c++17 lab3.cpp -o lab3.exe
//



#include <iostream>
#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

// Функция для последовательного вычисления среднего значения
double average_sequential(const int* arr, std::size_t n) {
    long long sum = 0; // long long, чтобы не было переполнения при большом N
    for (std::size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return static_cast<double>(sum) / static_cast<double>(n);
}

// Функция для параллельного вычисления среднего значения с OpenMP
double average_parallel(const int* arr, std::size_t n) {
    long long sum = 0;

#ifdef _OPENMP
    // Параллельный цикл for с редукцией по сумме:
    // каждый поток считает свою локальную sum, затем они складываются
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        sum += arr[i];
    }
#else
    // Если OpenMP не доступен, считаем последовательно
    for (std::size_t i = 0; i < n; ++i) {
        sum += arr[i];
    }
#endif

    return static_cast<double>(sum) / static_cast<double>(n);
}

int main() {
    using namespace std;

    // --- Настройки диапазона случайных чисел ---
    constexpr int RAND_MIN_VAL = 0;
    constexpr int RAND_MAX_VAL = 1000;

    cout << "Enter N (array size): ";
    size_t N;
    if (!(cin >> N) || N == 0) {
        cerr << "Invalid size\n";
        return 1;
    }

    // 1) Динамическое выделение памяти под массив
    //    new[] возвращает указатель на первый элемент
    int* a = new (nothrow) int[N];
    if (!a) {
        cerr << "Memory allocation failed\n";
        return 1;
    }

    // 2) Заполнение массива случайными числами
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);

    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(gen);
    }

    // 3) Подсчёт среднего значения

    // 3.1 Последовательный
    auto t1 = chrono::high_resolution_clock::now();
    double avg_seq = average_sequential(a, N);
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_seq = t2 - t1;

    // 3.2 Параллельный (OpenMP)
    auto t3 = chrono::high_resolution_clock::now();
    double avg_par = average_parallel(a, N);
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> dur_par = t4 - t3;

    cout << "Sequential average = " << avg_seq
         << ", time = " << dur_seq.count() << " ms\n";
    cout << "Parallel   average = " << avg_par
         << ", time = " << dur_par.count() << " ms\n";

    if (abs(avg_seq - avg_par) > 1e-9) {
        cerr << "Warning: averages differ (seq vs par)!\n";
    }

#ifdef _OPENMP
    cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#else
    cout << "OpenMP not available (compiled without -fopenmp)\n";
#endif

    // 4) Освобождение динамически выделенной памяти
    delete[] a;
    a = nullptr; // на всякий случай

    return 0;
}
