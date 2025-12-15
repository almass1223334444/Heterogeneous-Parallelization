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



    const int SIZE = 1'000'000;

    // Динамическое выделение памяти
    int* array = new int[SIZE];

    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Заполнение массива
    for (int i = 0; i < SIZE; i++) {
        array[i] = std::rand();
    }

    /* ================= ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ ================= */

    auto start_seq = std::chrono::high_resolution_clock::now();

    int min_seq = array[0];
    int max_seq = array[0];

    for (int i = 1; i < SIZE; i++) {
        if (array[i] < min_seq)
            min_seq = array[i];
        if (array[i] > max_seq)
            max_seq = array[i];
    }

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_seq = end_seq - start_seq;

    /* ================= ПАРАЛЛЕЛЬНАЯ ВЕРСИЯ (OpenMP) ================= */

    auto start_par = std::chrono::high_resolution_clock::now();

    int min_par = array[0];
    int max_par = array[0];

    #pragma omp parallel
    {
        int local_min = array[0];
        int local_max = array[0];

        #pragma omp for nowait
        for (int i = 0; i < SIZE; i++) {
            if (array[i] < local_min)
                local_min = array[i];
            if (array[i] > local_max)
                local_max = array[i];
        }

        #pragma omp critical
        {
            if (local_min < min_par)
                min_par = local_min;
            if (local_max > max_par)
                max_par = local_max;
        }
    }

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_par = end_par - start_par;

    /* ================= ВЫВОД РЕЗУЛЬТАТОВ ================= */

    std::cout << "Последовательная версия:" << std::endl;
    std::cout << "Минимум = " << min_seq << ", Максимум = " << max_seq << std::endl;
    std::cout << "Время: " << time_seq.count() << " мс\n" << std::endl;

    std::cout << "Параллельная версия (OpenMP):" << std::endl;
    std::cout << "Минимум = " << min_par << ", Максимум = " << max_par << std::endl;
    std::cout << "Время: " << time_par.count() << " мс" << std::endl;

    // Освобождение памяти
    delete[] array;

    return 0;
}
