#include <iostream>     // Ввод / вывод
#include <vector>       // Контейнер vector
#include <cstdlib>      // rand(), srand()
#include <ctime>        // time()
#include <chrono>       // Измерение времени
#include <omp.h>        // OpenMP
#ifdef _WIN32        // Проверка: если программа компилируется под Windows
#include <windows.h> // Подключение заголовка Windows API
#endif               // Конец условной компиляции

// ===================== ПОСЛЕДОВАТЕЛЬНЫЕ СОРТИРОВКИ =====================

// Сортировка пузырьком (последовательная)
void bubbleSort(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1])
                std::swap(a[j], a[j + 1]);
        }
    }
}

// Сортировка выбором (последовательная)
void selectionSort(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (a[j] < a[minIndex])
                minIndex = j;
        }
        std::swap(a[i], a[minIndex]);
    }
}

// Сортировка вставками (последовательная)
void insertionSort(std::vector<int>& a) {
    int n = a.size();
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// ===================== ПАРАЛЛЕЛЬНЫЕ СОРТИРОВКИ (OpenMP) =====================

// Пузырёк (параллельная версия)
void bubbleSortParallel(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j++) {
            if (a[j] > a[j + 1]) {
                std::swap(a[j], a[j + 1]);
            }
        }
    }
}

// Выбором (параллельная версия)
void selectionSortParallel(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            #pragma omp critical
            {
                if (a[j] < a[minIndex])
                    minIndex = j;
            }
        }
        std::swap(a[i], a[minIndex]);
    }
}

// Вставками (параллельная версия — учебная, неэффективная)
void insertionSortParallel(std::vector<int>& a) {
    int n = a.size();
    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

// ===================== ФУНКЦИЯ ТЕСТИРОВАНИЯ =====================

void testSort(void (*sortFunc)(std::vector<int>&),
              int size,
              const std::string& name)
{
    std::vector<int> arr(size);

    // Заполнение массива случайными числами
    for (int& x : arr)
        x = rand();

    // Замер времени
    auto start = std::chrono::high_resolution_clock::now();
    sortFunc(arr);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> time = end - start;

    std::cout << name
              << " | Размер: " << size
              << " | Время: " << time.count()
              << " мс" << std::endl;
}

// ===================== ГЛАВНАЯ ФУНКЦИЯ =====================

int main() {
    #ifdef _WIN32
    // Устанавливает кодировку UTF-8 для вывода в консоль Windows
    SetConsoleOutputCP(CP_UTF8);

    // Устанавливает кодировку UTF-8 для ввода с клавиатуры в консоль Windows
    SetConsoleCP(CP_UTF8);
#endif

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int sizes[] = {1000, 10000, 100000};

    for (int size : sizes) {
        std::cout << "\n===== Размер массива: " << size << " =====\n";

        testSort(bubbleSort, size, "Bubble (seq)");
        testSort(bubbleSortParallel, size, "Bubble (omp)");

        testSort(selectionSort, size, "Selection (seq)");
        testSort(selectionSortParallel, size, "Selection (omp)");

        testSort(insertionSort, size, "Insertion (seq)");
        testSort(insertionSortParallel, size, "Insertion (omp)");
    }

    return 0;
}
