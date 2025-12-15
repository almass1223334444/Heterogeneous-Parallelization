#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#ifdef _WIN32        // Проверка: если программа компилируется под Windows
#include <windows.h> // Подключение заголовка Windows API
#endif   

int main() {
     #ifdef _WIN32
    // Устанавливает кодировку UTF-8 для вывода в консоль Windows
    SetConsoleOutputCP(CP_UTF8);

    // Устанавливает кодировку UTF-8 для ввода с клавиатуры в консоль Windows
    SetConsoleCP(CP_UTF8);
#endif
    const int SIZE = 5'000'000;

    // Динамическое выделение памяти
    int* array = new int[SIZE];

    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Заполнение массива случайными числами от 1 до 100
    for (int i = 0; i < SIZE; i++) {
        array[i] = std::rand() % 100 + 1;
    }

    /* ================== ПОСЛЕДОВАТЕЛЬНЫЙ РАСЧЕТ СРЕДНЕГО ================== */

    auto start_seq = std::chrono::high_resolution_clock::now();

    long long sum_seq = 0;
    for (int i = 0; i < SIZE; i++) {
        sum_seq += array[i];
    }

    double average_seq = static_cast<double>(sum_seq) / SIZE;

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_seq = end_seq - start_seq;

    /* ================== ПАРАЛЛЕЛЬНЫЙ РАСЧЕТ СРЕДНЕГО (OpenMP) ================== */

    auto start_par = std::chrono::high_resolution_clock::now();

    long long sum_par = 0;

    // Параллельный цикл с reduction для корректного суммирования
    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < SIZE; i++) {
        sum_par += array[i];
    }

    double average_par = static_cast<double>(sum_par) / SIZE;

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_par = end_par - start_par;

    /* ================== ВЫВОД РЕЗУЛЬТАТОВ ================== */

    std::cout << "Последовательное среднее: " << average_seq
              << ", Время: " << time_seq.count() << " мс" << std::endl;

    std::cout << "Параллельное среднее (OpenMP): " << average_par
              << ", Время: " << time_par.count() << " мс" << std::endl;

    // Освобождение памяти
    delete[] array;

    return 0;
}
