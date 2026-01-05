// ============================================================================ // заголовок комментария
// Task 2: Array (10,000) -> min/max (sequential vs OpenMP) + timing comparison // описание задачи
// Требование: закомментировать каждую строку                                      // требование задания
// ============================================================================ // конец заголовка

#include <iostream>   // ввод/вывод (cout, cin)
#include <vector>     // std::vector
#include <random>     // генератор случайных чисел
#include <chrono>     // измерение времени
#include <limits>     // numeric_limits

#ifdef _OPENMP        // если компилируем с OpenMP
#include <omp.h>      // функции OpenMP (omp_get_max_threads и др.)
#endif                // конец условия OpenMP

int main() {                                                                 // точка входа программы
    using namespace std;                                                     // чтобы не писать std:: каждый раз

    const int N = 10000;                                                     // размер массива по заданию
    const int RAND_MIN_VAL = 1;                                              // минимум случайных значений
    const int RAND_MAX_VAL = 100;                                            // максимум случайных значений

    vector<int> a(N);                                                        // создаём массив из N элементов

    random_device rd;                                                        // источник энтропии для seed
    mt19937 gen(rd());                                                       // генератор mt19937
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);          // равномерное распределение [1..100]

    for (int i = 0; i < N; ++i) {                                            // цикл заполнения массива
        a[i] = dist(gen);                                                    // записываем случайное число
    }                                                                        // конец цикла заполнения

    cout << "TASK 2 — Min/Max search (Sequential vs OpenMP)\n";              // заголовок вывода
    cout << "Array size: " << N << "\n";                                     // печатаем размер массива
    cout << "Random range: [" << RAND_MIN_VAL << ".." << RAND_MAX_VAL << "]\n"; // печатаем диапазон

#ifdef _OPENMP                                                                // если OpenMP доступен
    cout << "OpenMP: enabled, max threads = " << omp_get_max_threads() << "\n"; // выводим доступные потоки
#else                                                                         // иначе
    cout << "OpenMP: NOT enabled (compiled without -fopenmp)\n";             // предупреждение
#endif                                                                        // конец условия OpenMP

    // -----------------------------                                             // разделитель
    // 1) SEQUENTIAL MIN/MAX                                                     // подпись части
    // -----------------------------                                             // разделитель

    int min_seq = numeric_limits<int>::max();                                // стартовое значение минимума (очень большое)
    int max_seq = numeric_limits<int>::lowest();                             // стартовое значение максимума (очень маленькое)

    auto t1 = chrono::high_resolution_clock::now();                          // старт таймера для sequential

    for (int i = 0; i < N; ++i) {                                            // последовательный проход по массиву
        if (a[i] < min_seq) min_seq = a[i];                                  // обновляем минимум
        if (a[i] > max_seq) max_seq = a[i];                                  // обновляем максимум
    }                                                                        // конец последовательного прохода

    auto t2 = chrono::high_resolution_clock::now();                          // конец таймера для sequential
    chrono::duration<double, milli> dur_seq = t2 - t1;                       // длительность sequential в мс

    // -----------------------------                                             // разделитель
    // 2) PARALLEL MIN/MAX (OpenMP)                                              // подпись части
    // -----------------------------                                             // разделитель

    int min_par = numeric_limits<int>::max();                                // стартовое значение минимума для parallel
    int max_par = numeric_limits<int>::lowest();                             // стартовое значение максимума для parallel

    auto t3 = chrono::high_resolution_clock::now();                          // старт таймера для parallel

#ifdef _OPENMP                                                                // если OpenMP доступен
    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par) schedule(static) // параллельный цикл с редукцией
    for (int i = 0; i < N; ++i) {                                            // параллельный проход по массиву
        if (a[i] < min_par) min_par = a[i];                                  // локально обновляем минимум (потом редукция)
        if (a[i] > max_par) max_par = a[i];                                  // локально обновляем максимум (потом редукция)
    }                                                                        // конец параллельного цикла
#else                                                                         // если OpenMP недоступен
    for (int i = 0; i < N; ++i) {                                            // fallback: обычный цикл
        if (a[i] < min_par) min_par = a[i];                                  // обновляем минимум
        if (a[i] > max_par) max_par = a[i];                                  // обновляем максимум
    }                                                                        // конец fallback цикла
#endif                                                                        // конец условия OpenMP

    auto t4 = chrono::high_resolution_clock::now();                          // конец таймера для parallel
    chrono::duration<double, milli> dur_par = t4 - t3;                       // длительность parallel в мс

    // -----------------------------                                             // разделитель
    // 3) OUTPUT + COMPARISON                                                    // подпись части
    // -----------------------------                                             // разделитель

    cout << "\nRESULTS\n";                                                   // печатаем заголовок результатов
    cout << "Sequential: min = " << min_seq << ", max = " << max_seq         // вывод sequential min/max
         << ", time = " << dur_seq.count() << " ms\n";                       // вывод sequential time
    cout << "Parallel:   min = " << min_par << ", max = " << max_par         // вывод parallel min/max
         << ", time = " << dur_par.count() << " ms\n";                       // вывод parallel time

    if (min_seq != min_par || max_seq != max_par) {                          // проверка корректности результатов
        cout << "WARNING: Results differ! (check OpenMP / logic)\n";         // сообщение об ошибке
    } else {                                                                 // иначе
        cout << "Check: OK (results match)\n";                               // результаты совпали
    }                                                                        // конец проверки

    double speedup = dur_seq.count() / dur_par.count();                      // ускорение (seq_time / par_time)

#ifdef _OPENMP                                                                // если OpenMP доступен
    cout << "Speedup: " << speedup << "x (using up to " << omp_get_max_threads() << " threads)\n"; // вывод ускорения
#else                                                                         // иначе
    cout << "Speedup: " << speedup << "x (OpenMP disabled -> parallel = sequential)\n"; // объяснение
#endif                                                                        // конец условия

    cout << "\nConclusion: For small arrays (N=10,000) parallel version may be similar or slower due to overhead.\n"; // вывод по смыслу

    return 0;                                                                // успешное завершение программы
}                                                                            // конец main
