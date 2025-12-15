#include <iostream>   // Ввод-вывод
#include <cstdlib>    // rand(), srand()
#include <chrono>     // Замер времени
#include <ctime>      // Подключение библиотеки для получения текущего времени (time)
#include <omp.h>      // Подключение библиотеки OpenMP для параллельных вычислений

#ifdef _WIN32        // Проверка: если программа компилируется под Windows
#include <windows.h> // Подключение заголовка Windows API
#endif               // Конец условной компиляции

int main() {

    #ifdef _WIN32
    // Устанавливает кодировку UTF-8 для вывода в консоль Windows
    SetConsoleOutputCP(CP_UTF8);

    // Устанавливает кодировку UTF-8 для ввода с клавиатуры в консоль Windows
    SetConsoleCP(CP_UTF8);
#endif

    // Размер динамического массива
    const int SIZE = 1'000'000;

    // Динамическое выделение памяти
    int* array = new int[SIZE];

    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Заполнение массива случайными числами
    for (int i = 0; i < SIZE; i++) {
        array[i] = std::rand();
    }

    // Начало замера времени
    auto start = std::chrono::high_resolution_clock::now();

    // Последовательный поиск минимума и максимума
    int minValue = array[0];
    int maxValue = array[0];

    for (int i = 1; i < SIZE; i++) {
        if (array[i] < minValue)
            minValue = array[i];

        if (array[i] > maxValue)
            maxValue = array[i];
    }

    // Конец замера времени
    auto end = std::chrono::high_resolution_clock::now();

    // Вычисление длительности в миллисекундах
    std::chrono::duration<double, std::milli> duration = end - start;

    // Вывод результатов
    std::cout << "Минимальный элемент: " << minValue << std::endl;
    std::cout << "Максимальный элемент: " << maxValue << std::endl;
    std::cout << "Время выполнения алгоритма (сравнение): "
              << duration.count() << " мс" << std::endl;

    // Освобождение памяти
    delete[] array;

    return 0;
}
