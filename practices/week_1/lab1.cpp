// lab1.cpp
// Работа с массивами + параллелизация OpenMP
// Программа:
// 1) Создаёт массив из N элементов и заполняет его случайными числами в диапазоне [RAND_MIN, RAND_MAX]
// 2) Находит минимальное и максимальное значение последовательно
// 3) Находит минимальное и максимальное значение параллельно с помощью OpenMP (reduction)
// 4) Измеряет и сравнивает время выполнения последовательного и параллельного алгоритмов
//
// Сборка (PowerShell):
// g++ -fopenmp -O2 -std=c++17 "c:\Programming_C++\Лабораторные работы\lab1.cpp" -o lab1.exe
// Запуск (пример):
// g++ -fopenmp -O2 -std=c++17 lab1.cpp -o lab1.exe
// $env:OMP_NUM_THREADS = '4'
// echo 1000000 | .\lab1.exe

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
	using namespace std;

	// --- Настройки диапазона случайных чисел ---
	// Измените эти константы, чтобы изменить диапазон значений в массиве
	// Важно: не называть переменные RAND_MAX, RAND_MIN — такие имена присутствуют в системных макросах.
	constexpr int RAND_MIN_VAL = 1;   // <-- менять здесь минимум диапазона
	constexpr int RAND_MAX_VAL = 1000000; // <-- менять здесь максимум диапазона

	// Порог вывода массива: если N <= PRINT_LIMIT, массив печатается на экран
	constexpr size_t PRINT_LIMIT = 100; // <-- менять, если хотите печатать большие массивы

	cout << "Enter N (array size): ";
	size_t N;
	if (!(cin >> N) || N == 0) {
		cerr << "Invalid size\n";
		return 1;
	}

	vector<int> a(N);

	// Заполнение массива случайными числами в диапазоне [RAND_MIN, RAND_MAX]
	// Для изменения диапазона — поменяйте значения RAND_MIN и RAND_MAX выше.
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);
	for (size_t i = 0; i < N; ++i) a[i] = dist(gen);

	// Печать массива только при небольшом N, чтобы не захламлять вывод
	if (N <= PRINT_LIMIT) {
		cout << "Array: ";
		for (auto v : a) cout << v << ' ';
		cout << '\n';
	} else {
		cout << "Array size " << N << " (not printed)\n";
	}

	// -------------------------
	// Последовательный алгоритм
	// Идея: проходим по всем элементам, поддерживая текущие min и max.
	// Время выполнения: O(N)
	// -------------------------
	auto t1 = chrono::high_resolution_clock::now();
	int min_seq = a[0];
	int max_seq = a[0];
	for (size_t i = 1; i < N; ++i) {
		if (a[i] < min_seq) min_seq = a[i];
		if (a[i] > max_seq) max_seq = a[i];
	}
	auto t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_seq = t2 - t1;

	// -------------------------
	// Параллельный алгоритм (OpenMP)
	// Используем директиву: #pragma omp parallel for reduction(min: ...) reduction(max: ...)
	// Каждый поток вычисляет локальный min/max, затем они редуцируются в глобальные значения.
	// Обратите внимание: для маленькой нагрузки (малое N или простая работа на элемент)
	// накладные расходы на создание потоков и синхронизацию могут превысить выигрыш.
	// -------------------------
	int min_par = a[0];
	int max_par = a[0];
	auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
	// Пример использования: задать количество потоков внешне через переменную окружения:
	// $env:OMP_NUM_THREADS = '4'
	#pragma omp parallel for reduction(min: min_par) reduction(max: max_par)
	for (int i = 1; i < static_cast<int>(N); ++i) {
		if (a[i] < min_par) min_par = a[i];
		if (a[i] > max_par) max_par = a[i];
	}
#else
	// Если OpenMP не доступен, выполняем ту же работу последовательно
	for (size_t i = 1; i < N; ++i) {
		if (a[i] < min_par) min_par = a[i];
		if (a[i] > max_par) max_par = a[i];
	}
#endif
	auto t4 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_par = t4 - t3;

	// Вывод результатов и времени
	cout << "Sequential: min = " << min_seq << ", max = " << max_seq
		 << ", time = " << dur_seq.count() << " ms\n";
	cout << "Parallel:   min = " << min_par << ", max = " << max_par
		 << ", time = " << dur_par.count() << " ms\n";

	if (min_seq != min_par || max_seq != max_par) {
		cerr << "Warning: sequential and parallel results differ!\n";
	}

#ifdef _OPENMP
	// Печать числа потоков, доступных OpenMP
	cout << "OpenMP threads (max): " << omp_get_max_threads() << '\n';
	// Если хотите программно задать число потоков, можно вызвать:
	// omp_set_num_threads(n);
#else
	cout << "OpenMP not available (compiled without -fopenmp)\n";
#endif

	return 0;
}