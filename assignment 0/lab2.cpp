// lab2.cpp
// ========================================================================
// РАБОТА СО СТРУКТУРАМИ ДАННЫХ И ПАРАЛЛЕЛИЗАЦИЯ OpenMP
// ========================================================================
// 
// Программа реализует три основные структуры данных с поддержкой 
// параллельного выполнения операций:
//
// 1) ОДНОСВЯЗНЫЙ СПИСОК (Single Linked List)
//    - Динамическая структура с использованием указателей
//    - Фиксированное время добавления/удаления в начало O(1)
//    - Поиск элемента требует O(n) операций
//    - Потокобезопасность обеспечена мьютексом
//
// 2) СТЕК (Stack) - LIFO (Last In First Out)
//    - Используется вектор с указателем на вершину
//    - Операции push/pop выполняются за O(1)
//    - Типичное применение: обратный порядок, вызовы функций
//
// 3) ОЧЕРЕДЬ (Queue) - FIFO (First In First Out)
//    - Реализована на основе вектора
//    - Добавление в конец O(1), удаление из начала O(1) с оптимизацией
//    - Применение: системы обработки задач, буферизация данных
//
// ========================================================================
// ПАРАЛЛЕЛИЗАЦИЯ:
// ========================================================================
// Программа сравнивает две подхода:
// - ПОСЛЕДОВАТЕЛЬНОЕ выполнение: один поток обрабатывает все операции
// - ПАРАЛЛЕЛЬНОЕ выполнение: несколько потоков работают одновременно
//
// Важно: Параллелизация эффективна только при достаточно большом N,
// так как создание потоков имеет накладные расходы.
//
// ========================================================================
// СБОРКА (PowerShell):
// ========================================================================
// g++ -fopenmp -O2 -std=c++17 "c:\Programming_C\Lab_Works\lab2.cpp" -o lab2.exe
//
// ========================================================================
// ЗАПУСК (пример):
// ========================================================================
// $env:OMP_NUM_THREADS = '4'        # Задать 4 потока
// .\lab2.exe                         # Запустить программу
// ========================================================================

#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ============================================
// 1. ОДНОСВЯЗНЫЙ СПИСОК (Single Linked List)
// ============================================
// Структура узла:
// - data: значение, хранящееся в узле
// - next: указатель на следующий узел (nullptr в конце списка)
//
// Особенности:
// - Добавление в начало: O(1)
// - Удаление из начала: O(1)
// - Поиск элемента: O(n)
// - Необходимо освобождение памяти в деструкторе
template <typename T>
class LinkedListNode {
public:
	T data;                    // Данные узла
	LinkedListNode* next;      // Указатель на следующий узел
	
	// Конструктор узла
	LinkedListNode(T val) : data(val), next(nullptr) {}
};

template <typename T>
class LinkedList {
private:
	LinkedListNode<T>* head;   // Указатель на первый узел списка
	mutex mtx;                 // Мьютекс для потокобезопасности
	                           // Защищает доступ к списку при параллельных операциях
	
public:
	// Конструктор: инициализирует пустой список
	LinkedList() : head(nullptr) {}
	
	// Деструктор: освобождает всю выделенную память
	// Процесс:
	// 1. Пока список не пуст:
	//    - Сохраняем указатель на текущий узел
	//    - Переходим к следующему узлу
	//    - Удаляем текущий узел
	~LinkedList() {
		while (head) {
			LinkedListNode<T>* temp = head;
			head = head->next;
			delete temp;
		}
	}
	
	// Добавление элемента в начало списка (O(1))
	// Процесс:
	// 1. Создаем новый узел с переданным значением
	// 2. Устанавливаем его next на текущий head
	// 3. Делаем новый узел новым head'ом
	// Lock_guard автоматически освобождает мьютекс при выходе из области
	void addFront(T val) {
		lock_guard<mutex> lock(mtx);  // Захватываем мьютекс
		LinkedListNode<T>* newNode = new LinkedListNode<T>(val);
		newNode->next = head;
		head = newNode;
		// Мьютекс автоматически освобождается здесь
	}
	
	// Удаление элемента с начала списка (O(1))
	// Процесс:
	// 1. Проверяем, не пуст ли список
	// 2. Если не пуст: сохраняем указатель на head
	// 3. Переходим head на следующий узел
	// 4. Удаляем сохраненный узел
	// Возвращает true если удаление успешно, false если список был пуст
	bool removeFront() {
		lock_guard<mutex> lock(mtx);
		if (!head) return false;  // Список пуст
		LinkedListNode<T>* temp = head;
		head = head->next;
		delete temp;
		return true;
	}
	
	// Поиск элемента в списке (O(n))
	// Процесс:
	// 1. Начинаем с head'а
	// 2. Проходим по всем узлам
	// 3. Если найдено значение, возвращаем true
	// 4. Если достигли конца списка, возвращаем false
	// Возвращает true если элемент найден, false в противном случае
	bool search(T val) {
		lock_guard<mutex> lock(mtx);
		LinkedListNode<T>* current = head;
		while (current) {
			if (current->data == val) return true;  // Элемент найден
			current = current->next;
		}
		return false;  // Элемент не найден
	}
	
	// Получение размера списка (O(n))
	// Процесс: проходим по всем узлам и считаем их
	// Возвращает количество элементов в списке
	int size() {
		lock_guard<mutex> lock(mtx);
		int count = 0;
		LinkedListNode<T>* current = head;
		while (current) {
			count++;
			current = current->next;
		}
		return count;
	}
	
	// Печать всех элементов списка в формате: elem1 -> elem2 -> ... -> nullptr
	// Полезно для отладки и визуализации содержимого списка
	void print() {
		lock_guard<mutex> lock(mtx);
		LinkedListNode<T>* current = head;
		cout << "List: ";
		while (current) {
			cout << current->data << " -> ";
			current = current->next;
		}
		cout << "nullptr\n";
	}
};

// ============================================
// 2. СТЕК (Stack) - LIFO структура
// ============================================
// Принцип работы: Last In First Out (последний вошел - первый вышел)
// Данные организованы в стек (как тарелки, поставленные одна на другую)
//
// Операции:
// - push(val): добавить элемент на вершину стека (O(1))
// - pop(): удалить элемент с вершины (O(1))
// - isEmpty(): проверка пустоты (O(1))
// - size(): получить количество элементов (O(1))
//
// Реальные применения:
// - Обратный порядок элементов
// - Вызов функций (стек вызовов)
// - Алгоритмы поиска в глубину (DFS)
// - Вычисление выражений в обратной польской нотации
template <typename T>
class Stack {
private:
	vector<T> data;  // Хранилище элементов стека
	mutex mtx;       // Мьютекс для потокобезопасности при параллельном доступе
	
public:
	// Добавление элемента в стек (O(1))
	// Процесс:
	// 1. Захватываем мьютекс для защиты от конкурентного доступа
	// 2. Добавляем элемент в конец вектора (это вершина стека)
	// 3. Мьютекс автоматически освобождается
	void push(T val) {
		lock_guard<mutex> lock(mtx);
		data.push_back(val);  // Добавляем на вершину стека
	}
	
	// Удаление элемента из стека (O(1))
	// Процесс:
	// 1. Проверяем, не пуст ли стек
	// 2. Если не пуст: удаляем последний элемент (вершину)
	// 3. Возвращаем true при успехе, false если стек был пуст
	bool pop() {
		lock_guard<mutex> lock(mtx);
		if (data.empty()) return false;  // Стек пуст
		data.pop_back();                 // Удаляем вершину стека
		return true;
	}
	
	// Проверка, пуст ли стек (O(1))
	// Возвращает true если стек не содержит элементов
	bool isEmpty() {
		lock_guard<mutex> lock(mtx);
		return data.empty();
	}
	
	// Получение количества элементов в стеке (O(1))
	// Возвращает текущее количество элементов
	int size() {
		lock_guard<mutex> lock(mtx);
		return data.size();
	}
	
	// Печать содержимого стека (от вершины к основанию)
	// Формат: вершина (top) слева, основание (bottom) справа
	void print() {
		lock_guard<mutex> lock(mtx);
		cout << "Stack (top to bottom): ";
		// Печатаем от конца к началу (от вершины к основанию)
		for (int i = data.size() - 1; i >= 0; --i) {
			cout << data[i] << " ";
		}
		cout << "\n";
	}
};

// ============================================
// 3. ОЧЕРЕДЬ (Queue) - FIFO структура
// ============================================
// Принцип работы: First In First Out (первый вошел - первый вышел)
// Работает как очередь в магазине: кто пришел первый, тот и выходит первый
//
// Операции:
// - enqueue(val): добавить элемент в конец очереди (O(1))
// - dequeue(): удалить элемент из начала (O(1) с оптимизацией)
// - isEmpty(): проверка пустоты (O(1))
// - size(): получить количество элементов (O(1))
//
// Реальные применения:
// - Системы обработки задач (очередь на печать)
// - Буферизация данных
// - Поиск в ширину (BFS)
// - Очередь обслуживания
template <typename T>
class Queue {
private:
	vector<T> data;         // Хранилище элементов очереди
	size_t front_idx;       // Индекс начала очереди (элемент, который удалим первым)
	mutex mtx;              // Мьютекс для потокобезопасности
	
public:
	// Конструктор: инициализируем пустую очередь
	Queue() : front_idx(0) {}
	
	// Добавление элемента в конец очереди (O(1))
	// Процесс:
	// 1. Захватываем мьютекс
	// 2. Добавляем элемент в конец вектора
	// 3. Этот элемент будет удален последним
	void enqueue(T val) {
		lock_guard<mutex> lock(mtx);
		data.push_back(val);  // Добавляем в конец очереди
	}
	
	// Удаление элемента из начала очереди (O(1) амортизированная)
	// Процесс:
	// 1. Проверяем, есть ли элементы в очереди
	// 2. Если есть: увеличиваем индекс front_idx (логическое удаление)
	// 3. Оптимизация: если очередь более чем наполовину пустая,
	//    перестраиваем вектор, удаляя использованные элементы
	// Возвращает true при успехе, false если очередь была пуста
	bool dequeue() {
		lock_guard<mutex> lock(mtx);
		if (front_idx >= data.size()) return false;  // Очередь пуста
		front_idx++;
		
		// Оптимизация памяти: если очередь больше половины пустая,
		// удаляем использованные элементы и перестраиваем вектор
		// Это предотвращает бесконечный рост памяти при множественных dequeue
		if (front_idx > data.size() / 2 && front_idx > 100) {
			data.erase(data.begin(), data.begin() + front_idx);
			front_idx = 0;  // Сбрасываем индекс после удаления
		}
		return true;
	}
	
	// Проверка, пуста ли очередь (O(1))
	// Возвращает true если нет активных элементов в очереди
	bool isEmpty() {
		lock_guard<mutex> lock(mtx);
		return front_idx >= data.size();
	}
	
	// Получение количества элементов в очереди (O(1))
	// Возвращает разницу между конечным индексом и начальным
	int size() {
		lock_guard<mutex> lock(mtx);
		return max(0, (int)data.size() - (int)front_idx);
	}
	
	// Печать всех элементов очереди в порядке от начала к концу
	// Элемент слева - первый удаляемый (front)
	// Элемент справа - последний добавленный (back)
	void print() {
		lock_guard<mutex> lock(mtx);
		cout << "Queue (front to back): ";
		// Печатаем элементы от front_idx до конца вектора
		for (size_t i = front_idx; i < data.size(); ++i) {
			cout << data[i] << " ";
		}
		cout << "\n";
	}
};

// ============================================
// ФУНКЦИИ ТЕСТИРОВАНИЯ И БЕНЧМАРКИНГА
// ============================================
// Эти функции выполняют бенчмарк структур данных, сравнивая:
// 1. Последовательное выполнение операций (одиночный поток)
// 2. Параллельное выполнение операций (несколько потоков OpenMP)
//
// Метрики:
// - Время выполнения в миллисекундах
// - Количество успешно обработанных элементов
// - Коэффициент ускорения (speedup) = время_последовательное / время_параллельное
// ============================================

// Тестирование ОДНОСВЯЗНОГО СПИСКА с параллелизацией
void testLinkedListParallel(int N, int threads) {
	cout << "\n";
	cout << "=================================================================\n";
	cout << "                  LINKED LIST (Single linked list)\n";
	cout << "=================================================================\n";
	cout << "Task: Add " << N << " elements to a linked list\n";
	cout << "Threads for parallel mode: " << threads << "\n";
	cout << "-----------------------------------------------------------------\n";

	LinkedList<int> list;

	// ========== SEQUENTIAL ==========
	cout << "\n[1] SEQUENTIAL ADD\n";
	cout << "    Description: single thread adds all elements sequentially\n";
	cout << "    Method: simple for-loop without parallelization\n";
	cout << "    Status: starting...\n";

	auto t1 = chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		list.addFront(i);
	}
	auto t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_seq = t2 - t1;

	int size_seq = list.size();
	cout << "    Done.\n";
	cout << "    - Elements added: " << size_seq << "\n";
	cout << "    - Time: " << dur_seq.count() << " ms\n";
	cout << "    - Rate: " << (size_seq / dur_seq.count()) << " elements/ms\n";

	LinkedList<int> listPar;

	// ========== PARALLEL ==========
	cout << "\n[2] PARALLEL ADD\n";
	cout << "    Description: " << threads << " threads add elements concurrently\n";
	cout << "    Method: #pragma omp parallel for (work distribution)\n";
	cout << "    Synchronization: mutex (to protect shared list)\n";
	cout << "    Status: starting...\n";

	auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		listPar.addFront(i);
	}
#else
	for (int i = 0; i < N; ++i) {
		listPar.addFront(i);
	}
#endif
	auto t4 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_par = t4 - t3;

	int size_par = listPar.size();
	cout << "    Done.\n";
	cout << "    - Elements added: " << size_par << "\n";
	cout << "    - Time: " << dur_par.count() << " ms\n";
	cout << "    - Rate: " << (size_par / dur_par.count()) << " elements/ms\n";

	// ========== ANALYSIS ==========
	cout << "\n[3] RESULTS (analysis)\n";
	double speedup = dur_seq.count() / dur_par.count();
	cout << "    - Speedup: " << speedup << "x\n";
	cout << "    - Efficiency: " << (speedup / threads * 100) << "%\n";
	if (speedup > 1.0) {
		cout << "    - Parallelization provided a speedup.\n";
	} else {
		cout << "    - Parallelization was slower (N might be too small).\n";
	}
	cout << "=================================================================\n";
}

void testStackParallel(int N, int threads) {
	cout << "\n";
	cout << "=================================================================\n";
	cout << "                             STACK (LIFO)\n";
	cout << "=================================================================\n";
	cout << "Task: Push " << N << " elements to the stack\n";
	cout << "Threads for parallel mode: " << threads << "\n";
	cout << "-----------------------------------------------------------------\n";
	cout << "Note: LIFO - last in, first out\n";

	Stack<int> stackSeq;

	// ========== SEQUENTIAL ==========
	cout << "\n[1] SEQUENTIAL PUSH\n";
	cout << "    Description: single thread pushes all elements\n";
	cout << "    Operation: push on top of the stack\n";
	cout << "    Status: starting...\n";

	auto t1 = chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		stackSeq.push(i);
	}
	auto t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_seq = t2 - t1;

	int size_seq = stackSeq.size();
	cout << "    Done.\n";
	cout << "    - Elements pushed: " << size_seq << "\n";
	cout << "    - Time: " << dur_seq.count() << " ms\n";
	cout << "    - Rate: " << (size_seq / dur_seq.count()) << " elements/ms\n";

	Stack<int> stackPar;

	// ========== PARALLEL ==========
	cout << "\n[2] PARALLEL PUSH\n";
	cout << "    Description: " << threads << " threads push elements concurrently\n";
	cout << "    Method: #pragma omp parallel for\n";
	cout << "    Synchronization: mutex (protect stack data)\n";
	cout << "    Status: starting...\n";

	auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		stackPar.push(i);
	}
#else
	for (int i = 0; i < N; ++i) {
		stackPar.push(i);
	}
#endif
	auto t4 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_par = t4 - t3;

	int size_par = stackPar.size();
	cout << "    Done.\n";
	cout << "    - Elements pushed: " << size_par << "\n";
	cout << "    - Time: " << dur_par.count() << " ms\n";
	cout << "    - Rate: " << (size_par / dur_par.count()) << " elements/ms\n";

	// ========== ANALYSIS ==========
	cout << "\n[3] RESULTS (analysis)\n";
	double speedup = dur_seq.count() / dur_par.count();
	cout << "    - Speedup: " << speedup << "x\n";
	cout << "    - Efficiency: " << (speedup / threads * 100) << "%\n";
	if (speedup > 1.0) {
		cout << "    - Parallelization provided a speedup.\n";
	} else {
		cout << "    - Parallelization was slower (N might be too small).\n";
	}
	cout << "=================================================================\n";
}

void testQueueParallel(int N, int threads) {
	cout << "\n";
	cout << "=================================================================\n";
	cout << "                             QUEUE (FIFO)\n";
	cout << "=================================================================\n";
	cout << "Task: Enqueue " << N << " elements to the queue\n";
	cout << "Threads for parallel mode: " << threads << "\n";
	cout << "-----------------------------------------------------------------\n";
	cout << "Note: FIFO - first in, first out\n";

	Queue<int> queueSeq;

	// ========== SEQUENTIAL ==========
	cout << "\n[1] SEQUENTIAL ENQUEUE\n";
	cout << "    Description: single thread enqueues all elements\n";
	cout << "    Status: starting...\n";

	auto t1 = chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		queueSeq.enqueue(i);
	}
	auto t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_seq = t2 - t1;

	int size_seq = queueSeq.size();
	cout << "    Done.\n";
	cout << "    - Elements enqueued: " << size_seq << "\n";
	cout << "    - Time: " << dur_seq.count() << " ms\n";
	cout << "    - Rate: " << (size_seq / dur_seq.count()) << " elements/ms\n";

	Queue<int> queuePar;

	// ========== PARALLEL ==========
	cout << "\n[2] PARALLEL ENQUEUE\n";
	cout << "    Description: " << threads << " threads enqueue elements concurrently\n";
	cout << "    Method: #pragma omp parallel for\n";
	cout << "    Synchronization: mutex (protect queue data)\n";
	cout << "    Status: starting...\n";

	auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		queuePar.enqueue(i);
	}
#else
	for (int i = 0; i < N; ++i) {
		queuePar.enqueue(i);
	}
#endif
	auto t4 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_par = t4 - t3;

	int size_par = queuePar.size();
	cout << "    Done.\n";
	cout << "    - Elements enqueued: " << size_par << "\n";
	cout << "    - Time: " << dur_par.count() << " ms\n";
	cout << "    - Rate: " << (size_par / dur_par.count()) << " elements/ms\n";

	// ========== ANALYSIS ==========
	cout << "\n[3] RESULTS (analysis)\n";
	double speedup = dur_seq.count() / dur_par.count();
	cout << "    - Speedup: " << speedup << "x\n";
	cout << "    • Efficiency: " << (speedup / threads * 100) << "%\n";
	if (speedup > 1.0) {
		cout << "    • Parallelization provided a speedup.\n";
	} else {
		cout << "    • Parallelization was slower (N might be too small).\n";
	}
	cout << "=================================================================\n";
}

void testCombined(int N, int threads) {
	cout << "\n";
	cout << "---------------------------------------------------------------\n";
	cout << "                COMBINED TEST (Mixed Operations)\n";
	cout << "---------------------------------------------------------------\n";
	cout << "Task: Perform " << N << " mixed operations\n";
	cout << "      (enqueue + dequeue)\n";
	cout << "Threads for parallel mode: " << threads << "\n";
	cout << "---------------------------------------------------------------\n";
	cout << "Description: Simulates a queue workload with concurrent producers\n";
	cout << "             and sequential consumers\n";
	
	Queue<int> queue;
	
	// ========== ПОСЛЕДОВАТЕЛЬНОЕ ВЫПОЛНЕНИЕ ==========
	cout << "\n[1] SEQUENTIAL OPERATIONS\n";
	cout << "    Description: single thread enqueues and dequeues elements\n";
	cout << "    Logic: enqueue each element; dequeue every second element\n";
	cout << "    Status: starting...\n";
	
	auto t1 = chrono::high_resolution_clock::now();
	for (int i = 0; i < N; ++i) {
		queue.enqueue(i);
		if (i % 2 == 0 && !queue.isEmpty()) {
			queue.dequeue();
		}
	}
	auto t2 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_seq = t2 - t1;
	
	cout << "    Done.\n";
	cout << "    - Remaining elements: " << queue.size() << "\n";
	cout << "    - Time: " << dur_seq.count() << " ms\n";
	cout << "    - Ops per ms: " << (N / dur_seq.count()) << "\n";
	
	Queue<int> queuePar;
	
	// ========== ПАРАЛЛЕЛЬНОЕ ВЫПОЛНЕНИЕ ==========
	cout << "\n[2] PARALLEL ENQUEUE OPERATIONS\n";
	cout << "    Description: " << threads << " threads enqueue elements concurrently\n";
	cout << "    Note: dequeue remains sequential due to FIFO semantics\n";
	cout << "    Status: starting...\n";
	
	auto t3 = chrono::high_resolution_clock::now();
#ifdef _OPENMP
	omp_set_num_threads(threads);
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		queuePar.enqueue(i);
	}
#else
	for (int i = 0; i < N; ++i) {
		queuePar.enqueue(i);
	}
#endif
	
	// Последовательное удаление (не параллелизуется из-за FIFO природы)
	cout << "    - Parallel enqueue finished, starting sequential dequeue...\n";
	for (int i = 0; i < N / 2; ++i) {
		if (!queuePar.isEmpty()) {
			queuePar.dequeue();
		}
	}
	auto t4 = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> dur_par = t4 - t3;
	
	cout << "    Done.\n";
	cout << "    - Remaining elements: " << queuePar.size() << "\n";
	cout << "    - Time: " << dur_par.count() << " ms\n";
	cout << "    - Ops per ms: " << (N / dur_par.count()) << "\n";
	
	// ========== АНАЛИЗ И СРАВНЕНИЕ ==========
	cout << "\n[3] RESULTS (analysis)\n";
	double speedup = dur_seq.count() / dur_par.count();
	cout << "    - Speedup: " << speedup << "x\n";
	cout << "    - Efficiency: " << (speedup / threads * 100) << "%\n";
	cout << "═══════════════════════════════════════════════════════════════\n";
}

int main() {
	cout << "\n";
	cout << "=================================================================\n";
	cout << "    DATA STRUCTURES BENCHMARK WITH OpenMP (C++)\n";
	cout << "=================================================================\n";

	// Get available threads
	int num_threads = 1;
#ifdef _OPENMP
	num_threads = omp_get_max_threads();
	cout << "INFO: OpenMP available\n";
	cout << "  - Available threads: " << num_threads << "\n";
	cout << "  - Compiled with -fopenmp\n";
#else
	cout << "INFO: OpenMP not available (compiled without -fopenmp)\n";
	cout << "  - Program will run with a single thread\n";
#endif

	cout << "\n";
	cout << "Benchmark will run the following tests:\n";
	cout << "  1) LinkedList (single linked list)\n";
	cout << "  2) Stack (LIFO)\n";
	cout << "  3) Queue (FIFO)\n";
	cout << "  4) Combined test (mixed operations)\n";
	cout << "\n";

	vector<int> test_sizes = {10000, 100000, 1000000};
	cout << "Test sizes: ";
	for (int i = 0; i < test_sizes.size(); ++i) {
		cout << test_sizes[i] << (i + 1 < test_sizes.size() ? ", " : "\n");
	}

	for (int N : test_sizes) {
		cout << "\n-----------------------------------------------------------------\n";
		cout << " Running benchmark round: N = " << N << "\n";
		cout << "-----------------------------------------------------------------\n";

		testLinkedListParallel(N, num_threads);
		testStackParallel(N, num_threads);
		testQueueParallel(N, num_threads);
		testCombined(N, num_threads);

		cout << "\nRound completed.\n";
	}

	// Demonstration
	cout << "\nDEMONSTRATION: basic operations\n";
	LinkedList<int> demoList;
	cout << "Adding elements to linked list: 5, 3, 7, 1\n";
	demoList.addFront(5);
	demoList.addFront(3);
	demoList.addFront(7);
	demoList.addFront(1);
	demoList.print();
	cout << "List size: " << demoList.size() << "\n";
	cout << "Search for 3: " << (demoList.search(3) ? "Found" : "Not found") << "\n";
	demoList.removeFront();
	demoList.print();

	cout << "\nStack demo:\n";
	Stack<int> demoStack;
	demoStack.push(10);
	demoStack.push(20);
	demoStack.push(30);
	demoStack.print();
	cout << "Stack size: " << demoStack.size() << "\n";
	demoStack.pop();
	demoStack.print();

	cout << "\nQueue demo:\n";
	Queue<int> demoQueue;
	demoQueue.enqueue(100);
	demoQueue.enqueue(200);
	demoQueue.enqueue(300);
	demoQueue.print();
	cout << "Queue size: " << demoQueue.size() << "\n";
	demoQueue.dequeue();
	demoQueue.print();

	cout << "\nAll done.\n";
	return 0;
}
